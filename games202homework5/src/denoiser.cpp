#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
// #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject, fill into m_accColor
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);
            int lastX, lastY;
            if (-1 == frameInfo.m_id(x, y)) { // ��������
                m_misc(x, y) = frameInfo.m_beauty(x, y);
                m_misc(x, y) = Float3(2, 0, 0);
            }
            else { // ������
                int itemId = frameInfo.m_id(x, y);
                auto& itemToWord = frameInfo.m_matrix[itemId];
                auto& itemWorldPos = frameInfo.m_position(x, y);
                Matrix4x4 matItemWorldPos;
                memset(matItemWorldPos.m, 0, sizeof(float) * 16);
                matItemWorldPos.m[0][3] = itemWorldPos.x;
                matItemWorldPos.m[1][3] = itemWorldPos.y;
                matItemWorldPos.m[2][3] = itemWorldPos.z;
                matItemWorldPos.m[3][3] = 1;

                // // ����������ǲ����������껹���������
                // auto curWorldToScreen = frameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
                // auto curScreenPos = curWorldToScreen * matItemWorldPos;
                // auto preCamToScreen = preWorldToScreen * Inverse(preWorldToCamera);

                auto& preItemToWord = m_preFrameInfo.m_matrix[itemId];

                // ���������Ļ�����Ӧ�ĵ㣬��һ���ǵ�ǰ���壬����һ�£���ǰ֡����һ֡����������
                // �㻹����Ķ����ڵ������ǵ�ǰ֡���ƶ�������
                // ע��lastScreePos��������ǣ�[x,y,depth,w]��û�о�����һ���ģ���Ҫ�Լ���һ����
                auto lastScreenPos = preWorldToScreen * // ����õ����Ļ����
                    preItemToWord *  // ������һ֡��������������
                    Inverse(itemToWord) *  // �����ϸõ�ı�������
                    matItemWorldPos; // ������������ض�Ӧ�ĵ����������

                lastX = lastScreenPos.m[0][3] / lastScreenPos.m[3][3];
                lastY = lastScreenPos.m[1][3] / lastScreenPos.m[3][3];
                if (0 <= lastX && lastX < width && 0 <= lastY && lastY < height) {
                    if (m_preFrameInfo.m_id(lastX, lastY) == itemId) {
                        m_misc(x, y) = m_preFrameInfo.m_beauty(lastX, lastY);
                        m_valid(x, y) = true;
                    }
                    else {
                        m_misc(x, y) = frameInfo.m_beauty(x, y);
                        // ������ԣ�
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
    
    // ����ɫ�����󣬲�ȥ����gaussģ��
    auto color1 = frameinfo.m_beauty(pixelPos1.x, pixelPos1.y);
    auto color2 = frameinfo.m_beauty(pixelPos2.x, pixelPos2.y);
    auto deltaColor = color1 - color2;
    
    // ƽ��ķ�������󣬲�ȥ���ף�eg: normal1 * normal2 = (0,0,0)
    // ��ô�ͻ�ʹ��deltaNormal = 1����Ȼ��˹�Ͳ�ȥ���ף���Ϊ3*sigmaNormal�ŵ���0.3
    auto normal1 = frameinfo.m_normal(pixelPos1.x, pixelPos1.y);
    auto normal2 = frameinfo.m_normal(pixelPos2.x, pixelPos2.y);
    auto deltaNormal = SafeAcos(Dot(normal1, normal2));

    // ��˼�ǿ��������ռ��������Զ��ƽ�棬�Ͳ�ȥ���ף������ṩ��һ�ֱ�ֻ�Ǽ򵥼���������ȵĲ�ֵ���õ�ָ��,
    // ���賡�������ߺ�ǽ��ƽ�У���ô�������ص���Ȼ�仯���룬����һ��kernel�ڱ�Ӧ�ù���
    // �����ص�û�й��ף���Ҳ�����жϣ��������ƽ��ƽ��ľ��볬����3*sigmaPlane���򲻻ṱ��
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

    // Reproject previous frame color to current
    if (m_useTemportal) {
        std::cout << "doing reprojection! " << std::endl;
        Reprojection(frameInfo);
        // TemporalAccumulation(filteredColor);
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
