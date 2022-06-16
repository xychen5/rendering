# rendering
渲染相关代码
---
title: realTimeRayTracing - games202实时渲染课后作业
date: 2022-06-06 14:56:42
tags:
- render
- gl
catogories:
- render
- gl
---

# 1 实时渲染里的光线追踪
## 1.1 为什么不用离线的蒙特卡洛呢？
回顾基于蒙特卡罗中的rayTracing，大致看一下其计算量：
当图像目标设置为768*768，然后每个像素sample32次（蒙特卡罗的经典方式），每次光线进入到下一层的概率为0.25，那么平均castray会被计算多少次呢？32 * （1*0.25 + 2 * 0.125 + 3 *0.0625 + ...) 大概就是0.75*32 = 24次，也就768*768*24的castRay的计算，花费的时间大概为23min，效果为：

![spp32_768x768_possiblity0](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/spp32_768x768_possiblity0.2njd3ot0e7u0.webp)
很显然这样子的计算速度在实时渲染里是太慢了，而且噪声依然很明显，于是我们考虑新的办法：时间上图像的连续性 + 滤波。


## 1.2 工业界的hack
注意到光线追踪是连续的两帧之间的光线追踪，那么可以假设当前帧的结果来自上一帧和当前帧的结合，也就是利用的时间上图像的连续性，具体的利用方式见下图：
![rtr_denoising&Filter](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/rtr_denoising%26Filter.1qcekmgz1134.webp)



# 2 具体的实时光线追踪降噪
## 2.1 单帧滤波实现
这里给出gauss滤波的实现: 参考提交：[Feature: 添加高斯模糊，sigma越大，中心权重越大，kernel大小和sigma要相互关联](https://github.com/xychen5/rendering/commit/371251639f2f1a427ce7f564375e82dfdccc1c71)
```python
// -------------------------------- 算法本身
For each pixel i
    sum_of_weights = sum_of_weighted_values = 0.0
    For each pixel j around i
        Calculate the weight w_ij = G(|i - j|, sigma)
        sum_of_weighted_values += w_ij * C^{input}[j]
        sum_of_weights += w_ij
    C^{output}[I] = sum_of_weighted_values / sum_of_weights
```

简单说明一下gauss滤波的效果，其实主要就是mean和deviation，下面的mean就是二维向量i :(x,y),deviation就是sigmaCoord，考虑到3sigma时高斯值降为0，那么为了让高斯模糊的效果不那么明显，我们将sigma = kernel / 3 / 2，将高斯的能量集中在i像素附近
```cpp
return pow(
        2.718281,
        -(sqrt(Dot(Abs(i - j), Abs(i - j))) / (2 * sigmaCoord * sigmaCoord)) \
    );
```
很显然，kernel越大，越模糊，这个就不展示了，主要看一下sigma的影响：sigma越小，肯定模糊就越少：kenerl都是5，从
| sig: 5/6 | sig: 5/3 |
| - | - |
| ![sigmaCoord5div6kernel5](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/sigmaCoord5div6kernel5.3nwcedqo6la0.webp) | ![GausssigmaCoord5div3kernel5](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/GausssigmaCoord5div3kernel5.3q2fclg5na40.webp) |

给出双边滤波函数为：
$$
J(i,j) 
=\exp 
    \left(
        -{\frac {\| i-j \|^{2}}{2\sigma_p ^{2}}}
        -{\frac {\| \tilde{C}[i]-\tilde{C}[j] \|^{2}}{2\sigma_c ^{2}}}
        -{\frac {D_{normal}(i, j)^{2}}{2\sigma_n ^{2}}}
        -{\frac {D_{plane}(i, j)^{2}}{2\sigma_d ^{2}}}
    \right)
$$
其中的i,j表示两个像素位置，
- 1 $\sigma_p$，表示根据位置做高斯模糊的高斯分布半径，我取的是kernelSize / 3.
- 2 $\sigma_c$，表示两个位置的颜色的差距，太大了就不参与模糊，那么颜色范围则取0.6吧，对应的是下面gif中灯具的周围变清晰的变化
- 3 $\sigma_n$，代表衡量两个点的法向的区别，其中$D_{normal}(i,j) = arccos(Normal[i] \cdot Normal[j])$，很显然，当两个法向接近的时候，就会使得该项为0，也就是不会被排除掉，当然如果两个反向垂直，那么会被忽略，对应的是下面gif中墙面连接处的变化
- 4 $\sigma_d$，代表衡量两个像素所在平面之间的差距，$D_{plane}(i,j) = Normal[i] \cdot (\frac {Position[i] - Position[j]}{\| Position[i] - Position[j] \|})$，意思是考虑两个空间中相隔较远的平面，就不去贡献！这种提供了一种比只是简单计算两个深度的差值更好的指标， 假设场景的视线和墙面平行，那么相邻像素的深度会变化距离，导致一个kernel内本应该贡献的像素点没有贡献，这也就是判断，如果两个平行平面的距离超过了3*sigmaPlane，则不会贡献


这里给出滤波后的结果(四种变化分别代表着从1到4增量累加的效果)：
![boxJointFilter](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/boxJointFilter.3xsl50bh0g00.gif)

实现代码如下：
```cpp
float dobuleJointFilter(
    Float3 pixelPos1,
    Float3 pixelPos2,

    float sigmaPlane, 
    float sigmaColor, 
    float sigmaNormal, 
    float sigmaCoord,
    const FrameInfo& frameinfo) {
    auto deltaPos = pixelPos1 - pixelPos2;
    
    // 当颜色差距过大，不去贡献gauss模糊
    auto color1 = frameinfo.m_beauty(pixelPos1.x, pixelPos1.y);
    auto color2 = frameinfo.m_beauty(pixelPos2.x, pixelPos2.y);
    auto deltaColor = color1 - color2;
    
    // 平面的法向差距过大，不去贡献：eg: normal1 * normal2 = (0,0,0)
    // 那么就会使得deltaNormal = 1，自然高斯就不去贡献，因为3*sigmaNormal才等于0.3
    auto normal1 = frameinfo.m_normal(pixelPos1.x, pixelPos1.y);
    auto normal2 = frameinfo.m_normal(pixelPos2.x, pixelPos2.y);
    auto deltaNormal = SafeAcos(Dot(normal1, normal2));

    // 意思是考虑两个空间中相隔较远的平面，就不去贡献！这种提供了一种比只是简单计算两个深度的差值更好的指标,
    // 假设场景的视线和墙面平行，那么相邻像素的深度会变化距离，导致一个kernel内本应该贡献
    // 的像素点没有贡献，这也就是判断，如果两个平行平面的距离超过了3*sigmaPlane，则不会贡献
    auto pos3D1 = frameinfo.m_position(pixelPos1.x, pixelPos1.y);
    auto pos3D2 = frameinfo.m_position(pixelPos2.x, pixelPos2.y);
    auto deltaPos3D = pos3D1 - pos3D2;
    Float3 deltaPlane(0, 0, 0);
    if (0 != Dot(deltaPos3D, deltaPos3D)) {
        deltaPlane = normal1 * (deltaPos3D / sqrt(Dot(deltaPos3D, deltaPos3D)));
    }
    return pow(
        2.718281,
        -(Dot(deltaPos, deltaPos) / (2 * sigmaCoord * sigmaCoord)) \
        -(Dot(deltaColor, deltaColor) / (2 * sigmaColor * sigmaColor)) \
        -(Dot(deltaNormal, deltaNormal) / (2 * sigmaNormal * sigmaNormal)) \
        -(Dot(deltaPlane, deltaPlane) / (2 * sigmaPlane * sigmaPlane)) 
    );
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 7;
    m_sigmaCoord = static_cast<float>(kernelRadius) / 3.0;
    m_sigmaPlane = 0.35;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter
            double weightsSum = 0;
            double c1ValueSum = 0;
            double c2ValueSum = 0;
            double c3ValueSum = 0;
            for (int newX = x - kernelRadius; newX <= x + kernelRadius; ++newX) {
                for (int newY = y - kernelRadius; newY <= y + kernelRadius; ++newY) {
                    int cx = std::min(std::max(0, newX), width);
                    int cy = std::min(std::max(0, newY), height);
                    float weight = dobuleJointFilter(
                        Float3(x, y, 0), Float3(cx, cy, 0), 
                        m_sigmaPlane, m_sigmaColor, m_sigmaNormal, m_sigmaCoord,
                        frameInfo
                    );
                    
                    // std::cout << "pixel pos: " << cx << ", " << cy << " | " << frameInfo.m_beauty(cx, cy).x << " \ "
                    //     << frameInfo.m_beauty(cx, cy).y  << " \ "
                    //     << frameInfo.m_beauty(cx, cy).z << std::endl;

                    weightsSum += weight;
                    c1ValueSum += weight * frameInfo.m_beauty(cx, cy).x;
                    c2ValueSum += weight * frameInfo.m_beauty(cx, cy).y;
                    c3ValueSum += weight * frameInfo.m_beauty(cx, cy).z;
                }
            }
            c1ValueSum /= weightsSum;
            c2ValueSum /= weightsSum;
            c3ValueSum /= weightsSum;
            filteredImage(x, y) = Float3(c1ValueSum, c2ValueSum, c3ValueSum);
        }
    }
    return filteredImage;
}


```

## 2.2 投影上一帧结果
计算当前帧每个像素在上一帧的对应点，并将上一帧的结果投影到当前帧。

### 2.1 MVP转化和viewPort转化：
- 1 首先是MVP: M，物体从世界坐标000旋转平移缩放，V，将相机放在合适的位置的转换：比如说放到000，P，投影转换，将物体从世界坐标转到边长为-1到1的一个cur里面，可以做视锥剔除，一般是跟你的透视矩阵定义的有关，比如 near，far，bottom，up，fov几个参数，做完透视投影以后，我们认为物体的坐标都在NDC里面了
![PerspectiveProjections](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/PerspectiveProjections.4vs3msb3g5u0.webp)
- 2 viewPort投影（视口投影）
- 2.0 主要就是将NDC空间里的值，切换成像素坐标
- 2.1 将物体坐标放缩到图片大小: Scale the (-1,-1) to (+1,+1) viewing window to the image’s width and height.
- 2.2 将near平面(这也是成像平面)左下角的坐标从（-w/2,-h/2)移动到0,0: Offset the lower-left corner at (-width/2,-height/2) to the image’s origin.
![Viewports](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/Viewports.7d4mhekdzr00.webp)

### 2.2 从上一帧获取结果
主要思路就是：
- 1 计算当前帧每个像素在上一帧的对应点，并将上一帧的结果投影到当前帧。

那么当前帧的一个物体的世界坐标如何投影到上一帧对应的像素坐标呢？
$Screen_{i-1} = P_{i-1}V_{i-1}M_{i}M_{i}^{-1}World_i$
看下面从matItemWorldPos转换到上一帧的屏幕坐标经历的过程：
```cpp
                // 注意lastScreePos算出来的是，[x,y,depth,w]，没有经过归一化的，需要自己归一化啊
                auto lastScreenPos = preWorldToScreen * // 物体该点的屏幕坐标
                    preItemToWord *  // 物体上一帧这个点的世界坐标
                    Inverse(itemToWord) *  // 物体上该点的本地坐标
                    matItemWorldPos; // 物体上这个像素对应的点的世界坐标
                lastX = lastScreenPos.m[0][3] / lastScreenPos.m[3][3];
                lastY = lastScreenPos.m[1][3] / lastScreenPos.m[3][3];
```

- 2 那么用几个简单标志判断上一帧是否有效： 算出来的屏幕坐标对应的点，不一定是当前物体，想象一下，当前帧的上一帧，物体的这个点还被别的东西遮挡，但是当前帧则移动出来了
  - 2.1 上一帧的点是否在屏幕内部(用蓝色表示)
  - 2.2 上一帧的点他们是否为同一个物体(用绿色表示)
  - 2.3 上一帧的点是否对应的是物体(不是的话用红色表示)
- 3 只有同时满足2的3个要求，我们认为这是有用的，否则直接用当前帧对应的像素点即可
- 4 并将投影是否合法保存在 m_valid 以供我们在累积多帧信息时使用，因为存在macc_color是上一帧的结果，我们用m_valid(x,y)记录上一帧的像素在当前帧是否任然可以使用，不能的话就用当前帧滤波以后的结果就行

这里给出backProjection的代码：
```cpp
// 主要目的：1. 得到当前帧的像素x,y在上一帧中是否可用，存在valid(x,y)上，m_accColor存的上一帧的信息就不要动了
// 主要目的：那些不valid点，就不会进入accumulateColor的过程，转而直接使用当前帧滤波以后的结果
void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject, fill into m_accColor
            m_valid(x, y) = false; // 这个valid标志，会被时间上的累计中的clamp干掉
            // m_misc(x, y) = Float3(0.f);
            int lastX, lastY;
            if (-1 == frameInfo.m_id(x, y)) { // 不是物体
                m_misc(x, y) = frameInfo.m_beauty(x, y);
                // m_misc(x, y) = Float3(2, 0, 0);
            }
            else { // 是物体
                int itemId = frameInfo.m_id(x, y);
                auto& itemToWord = frameInfo.m_matrix[itemId];
                auto& itemWorldPos = frameInfo.m_position(x, y);
                Matrix4x4 matItemWorldPos;
                memset(matItemWorldPos.m, 0, sizeof(float) * 16);
                matItemWorldPos.m[0][3] = itemWorldPos.x;
                matItemWorldPos.m[1][3] = itemWorldPos.y;
                matItemWorldPos.m[2][3] = itemWorldPos.z;
                matItemWorldPos.m[3][3] = 1;

                // // 检测存的坐标是不是世界坐标还是相机坐标
                // auto curWorldToScreen = frameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
                // auto curScreenPos = curWorldToScreen * matItemWorldPos;
                // auto preCamToScreen = preWorldToScreen * Inverse(preWorldToCamera);

                auto& preItemToWord = m_preFrameInfo.m_matrix[itemId];

                // 算出来的屏幕坐标对应的点，不一定是当前物体，想象一下，当前帧的上一帧，物体的这个
                // 点还被别的东西遮挡，但是当前帧则移动出来了
                // 注意lastScreePos算出来的是，[x,y,depth,w]，没有经过归一化的，需要自己归一化啊
                auto lastScreenPos = preWorldToScreen * // 物体该点的屏幕坐标
                    preItemToWord *  // 物体上一帧这个点的世界坐标
                    Inverse(itemToWord) *  // 物体上该点的本地坐标
                    matItemWorldPos; // 物体上这个像素对应的点的世界坐标

                lastX = lastScreenPos.m[0][3] / lastScreenPos.m[3][3];
                lastY = lastScreenPos.m[1][3] / lastScreenPos.m[3][3];
                if (0 <= lastX && lastX < width && 0 <= lastY && lastY < height) {
                    if (m_preFrameInfo.m_id(lastX, lastY) == itemId) {
                        // 将上一帧的结果能否可以投影到当前帧记作m_valid(x,y)
                        // 然后将上一帧的颜色投影到当前帧，使用m_misc作为中间变量
                        // 最后还是存在m_accColor
                        // m_misc(x, y) = m_preFrameInfo.m_beauty(lastX, lastY); 
                        m_misc(x, y) = m_accColor(lastX, lastY); 
                        m_valid(x, y) = true;
                    }
                    else {
                        m_misc(x, y) = frameInfo.m_beauty(x, y);
                        // 方便调试：
                        // m_misc(x, y) = Float3(0, 2, 0);
                    }
                }
                else {
                    m_misc(x, y) = frameInfo.m_beauty(x, y);
                    // m_misc(x, y) = Float3(0, 0, 2);
                }
            }
        }
    }
    std::swap(m_misc, m_accColor);
}

```

- 4 结果展示：
这是带有debug信息的展示：
![boxReprojection](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/boxReprojection.791ps95hu840.gif)

![pinkRoomBackprojection5](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/pinkRoomBackprojection5.1zx84khtcsm8.gif)


## 2.3 累计多帧信息
在这个部分，你需要将已经降噪的当前帧图像 $\overline{C_{i}}$，与已经降噪的上一帧图像
$\overline{C_{i-1}}$ 进行结合，公式如下：
$$
\overline{C_{i-1}} \leftarrow \alpha\overline{C_{i}} +  (1 - \alpha) Clamp(\overline{C_{i-1}})
$$
其实比较好理解：
- 1 当上一帧的像素在当前帧对应的物体不是同一个物体，则阿尔法等于1
- 2 对于Clamp部分，我们首先需要计算 Ci 在 7×7 的邻域内的均值 µ 和方差 σ，
然后我们将上一帧的颜色 Ci−1Clamp 在 (µ − kσ, µ + kσ) 范围内。

# 最终结果展示
![boxFinalRes](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/boxFinalRes.4w0ddz6f8m00.gif)

针对pinkroom场景，需要改一下滤波器的color部分的deviation也就是$sigma_{color}$，最关键的是需要明白这个sigma如何取因为对于pinkroom，可以看到它颜色的变化很大:(上面是box,下面是pinkRoom)
![box](https://raw.githubusercontent.com/xychen5/blogImgs/main/imgs/book.7ckxjjg7dgs0.webp)

![pinkRoom](https://cdn.jsdelivr.net/gh/xychen5/blogImgs@main/imgs/pinkRoom.nyra8p6v9pc.webp)

所以我们调整simga到: 3*sigma <= meanDis 约等于：6，所以sigma我们取4.5左右就好

# Ref：
- 1 b站 202 高质量实时渲染
- 2 实现的代码： https://github.com/xychen5/rendering.git
