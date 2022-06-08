#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

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
            // TODO: Reproject
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp
            Float3 color = m_accColor(x, y);
            // TODO: Exponential moving average
            float alpha = 1.0f;
            m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
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
    return filteredColor;

    // Reproject previous frame color to current
    if (m_useTemportal) {
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
