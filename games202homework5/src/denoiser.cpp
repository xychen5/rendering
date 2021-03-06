#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

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

// 对lastFrame的xy处的像素clamp到u +- k*sigma, u是颜色均值，sigma是方差
Float3 TemporalClamp(
    const Buffer2D<Float3>& curFrame,
    const Buffer2D<Float3>& preFrame,
    int w, int h,
    int x, int y,
    int kernelRadius,
    float k
) {
    Float3 sum(0, 0, 0);
 
	Float3 accum(0, 0, 0);
    int n = pow(2 * kernelRadius + 1, 2);
    for (int newX = x - kernelRadius; newX <= x + kernelRadius; ++newX) {
        for (int newY = y - kernelRadius; newY <= y + kernelRadius; ++newY) {
            int cx = std::min(std::max(0, newX), w);
            int cy = std::min(std::max(0, newY), h);
            sum.x += curFrame(cx,cy).x;
            sum.y += curFrame(cx,cy).y;
            sum.z += curFrame(cx,cy).z;
        }
    }
	Float3 mean(sum.x / n, sum.y / n, sum.z / n); //均值
    
    for (int newX = x - kernelRadius; newX <= x + kernelRadius; ++newX) {
        for (int newY = y - kernelRadius; newY <= y + kernelRadius; ++newY) {
            int cx = std::min(std::max(0, newX), w);
            int cy = std::min(std::max(0, newY), h);
            accum.x += pow(curFrame(cx, cy).x - mean.x, 2);
            accum.y += pow(curFrame(cx, cy).y - mean.y, 2);
            accum.z += pow(curFrame(cx, cy).z - mean.z, 2);
        }
    }
    Float3 sigma( 
        std::sqrt(accum.x / (n - 1)),
        std::sqrt(accum.y / (n - 1)),
        std::sqrt(accum.z / (n - 1))
    ); // 方差

    Float3 st(mean.x - sigma.x, mean.y - sigma.y, mean.z - sigma.z);
    Float3 ed(mean.x + sigma.x, mean.y + sigma.y, mean.z + sigma.z);
    return Clamp(preFrame(x, y), st, ed);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 7;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp, 可以去除滤波漏下的outlier
            Float3 clampedColor = TemporalClamp(
                curFilteredColor,
                m_accColor,
                width, height,
                x, y,
                kernelRadius, 0.2
            );

            // TODO: Exponential moving average
            float alpha = 0.05;
            if (!m_valid(x, y)) { // 上一帧的x，y是不可以使用的
                alpha = 1;
            }
            m_misc(x, y) = Lerp(clampedColor, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}
    
// float m_alpha = 0.2f;
// float m_sigmaPlane = 0.1f;
// float m_sigmaColor = 0.6f; 
// float m_sigmaNormal = 0.1f; //
// float m_sigmaCoord = kenelSize / 3 / 2;
// float m_colorBoxK = 1.0f;
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
    int kernelRadius = 8;
    m_sigmaCoord = static_cast<float>(kernelRadius) / 2.0;
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

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        std::cout << "doing reprojection! " << std::endl;
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
