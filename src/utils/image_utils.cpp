#include "rknn_cpp/utils/image_utils.h"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cmath>
#include <fstream>

// OpenCV headers for image I/O
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace rknn_cpp
{
namespace utils
{

// 计算图像缓冲区大小
int getImageSize(const image_buffer_t& image)
{
    switch (image.format)
    {
        case ImageFormat::GRAY8:
            return image.width * image.height;
        case ImageFormat::RGB888:
            return image.width * image.height * 3;
        case ImageFormat::RGBA8888:
            return image.width * image.height * 4;
        case ImageFormat::YUV420SP_NV12:
        case ImageFormat::YUV420SP_NV21:
            return image.width * image.height * 3 / 2;
        default:
            return 0;
    }
}

// 创建图像缓冲区
image_buffer_t createImageBuffer(int width, int height, ImageFormat format)
{
    image_buffer_t image{};
    image.width = width;
    image.height = height;
    image.width_stride = width;
    image.height_stride = height;
    image.format = format;
    image.size = getImageSize(image);
    image.fd = -1;

    if (image.size > 0)
    {
        image.virt_addr = static_cast<unsigned char*>(malloc(image.size));
        if (image.virt_addr == nullptr)
        {
            std::cerr << "Failed to allocate " << image.size << " bytes for image buffer" << std::endl;
            image.size = 0;
        }
        else
        {
            memset(image.virt_addr, 0, image.size);
        }
    }

    return image;
}

// 释放图像缓冲区内存
void freeImage(image_buffer_t& image)
{
    if (image.virt_addr != nullptr)
    {
        free(image.virt_addr);
        image.virt_addr = nullptr;
    }
    image.size = 0;
    image.width = 0;
    image.height = 0;
}

// 复制图像缓冲区
image_buffer_t cloneImageBuffer(const image_buffer_t& src)
{
    image_buffer_t dst = createImageBuffer(src.width, src.height, src.format);
    if (dst.virt_addr != nullptr && src.virt_addr != nullptr)
    {
        memcpy(dst.virt_addr, src.virt_addr, src.size);
        dst.width_stride = src.width_stride;
        dst.height_stride = src.height_stride;
    }
    return dst;
}

// 打印图像信息
void printImageInfo(const image_buffer_t& image, const std::string& name)
{
    const char* format_names[] = {"GRAY8", "RGB888", "RGBA8888", "YUV420SP_NV21", "YUV420SP_NV12"};
    int format_idx = static_cast<int>(image.format);

    std::cout << "=== " << name << " Info ===" << std::endl;
    std::cout << "  Size: " << image.width << "x" << image.height << std::endl;
    std::cout << "  Stride: " << image.width_stride << "x" << image.height_stride << std::endl;
    std::cout << "  Format: " << (format_idx < 5 ? format_names[format_idx] : "UNKNOWN") << std::endl;
    std::cout << "  Buffer size: " << image.size << " bytes" << std::endl;
    std::cout << "  Virtual addr: " << static_cast<void*>(image.virt_addr) << std::endl;
}

#ifdef USE_OPENCV
// OpenCV版本的图像读取
bool readImageOpenCV(const std::string& path, image_buffer_t& image)
{
    try
    {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cerr << "Failed to read image: " << path << std::endl;
            return false;
        }

        // 转换颜色格式从BGR到RGB
        cv::Mat rgb_img;
        cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

        int width = rgb_img.cols;
        int height = rgb_img.rows;
        int channels = rgb_img.channels();
        int size = width * height * channels;

        std::cout << "Loaded image: " << width << "x" << height << "x" << channels << " from " << path << std::endl;

        // 设置图像数据
        if (image.virt_addr != nullptr && image.size >= size)
        {
            memcpy(image.virt_addr, rgb_img.data, size);
        }
        else
        {
            // 重新分配内存
            if (image.virt_addr != nullptr)
            {
                free(image.virt_addr);
            }
            image.virt_addr = static_cast<unsigned char*>(malloc(size));
            if (image.virt_addr == nullptr)
            {
                std::cerr << "Failed to allocate " << size << " bytes" << std::endl;
                return false;
            }
            memcpy(image.virt_addr, rgb_img.data, size);
        }

        image.width = width;
        image.height = height;
        image.width_stride = width;
        image.height_stride = height;
        image.size = size;
        image.fd = -1;

        // 根据通道数设置格式
        if (channels == 4)
        {
            image.format = ImageFormat::RGBA8888;
        }
        else if (channels == 1)
        {
            image.format = ImageFormat::GRAY8;
        }
        else
        {
            image.format = ImageFormat::RGB888;
        }

        return true;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "OpenCV exception: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Unknown exception occurred" << std::endl;
        return false;
    }
}
#endif

// 读取原始数据文件
bool readImageRaw(const std::string& path, image_buffer_t& image)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (image.virt_addr != nullptr && image.size >= static_cast<int>(file_size))
    {
        // 使用现有缓冲区
        if (!file.read(reinterpret_cast<char*>(image.virt_addr), file_size))
        {
            std::cerr << "Failed to read file: " << path << std::endl;
            return false;
        }
    }
    else
    {
        // 重新分配缓冲区
        if (image.virt_addr != nullptr)
        {
            free(image.virt_addr);
        }
        image.virt_addr = static_cast<unsigned char*>(malloc(file_size));
        if (image.virt_addr == nullptr)
        {
            std::cerr << "Failed to allocate " << file_size << " bytes" << std::endl;
            return false;
        }

        if (!file.read(reinterpret_cast<char*>(image.virt_addr), file_size))
        {
            std::cerr << "Failed to read file: " << path << std::endl;
            free(image.virt_addr);
            image.virt_addr = nullptr;
            return false;
        }
    }

    image.size = static_cast<int>(file_size);
    return true;
}

// 读取图像文件
bool readImage(const std::string& path, image_buffer_t& image)
{
    if (path.empty())
    {
        std::cerr << "Empty path provided" << std::endl;
        return false;
    }

    // 获取文件扩展名
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos)
    {
        std::cerr << "Missing file extension in: " << path << std::endl;
        return false;
    }

    std::string ext = path.substr(dot_pos);

    // 转换为小写
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // 根据扩展名选择读取方式
    if (ext == ".data")
    {
        return readImageRaw(path, image);
    }
    else
    {
#ifdef USE_OPENCV
        return readImageOpenCV(path, image);
#else
        std::cerr << "OpenCV not available, cannot read image: " << path << std::endl;
        return false;
#endif
    }
}

// 双线性插值的核心函数
bool cropAndScaleRGB(const unsigned char* src, int src_width, int src_height, int crop_x, int crop_y, int crop_width,
                     int crop_height, unsigned char* dst, int dst_width, int /*dst_height*/, int dst_box_x,
                     int dst_box_y, int dst_box_width, int dst_box_height)
{
    if (src == nullptr || dst == nullptr)
    {
        std::cerr << "Null pointer provided to cropAndScaleRGB" << std::endl;
        return false;
    }

    const int channels = 3;
    float x_ratio = static_cast<float>(crop_width) / static_cast<float>(dst_box_width);
    float y_ratio = static_cast<float>(crop_height) / static_cast<float>(dst_box_height);

    // 双线性插值缩放
    for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++)
    {
        for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++)
        {
            int dst_x_offset = dst_x - dst_box_x;
            int dst_y_offset = dst_y - dst_box_y;

            float src_x_f = dst_x_offset * x_ratio + crop_x;
            float src_y_f = dst_y_offset * y_ratio + crop_y;

            int src_x = static_cast<int>(src_x_f);
            int src_y = static_cast<int>(src_y_f);

            float x_diff = src_x_f - src_x;
            float y_diff = src_y_f - src_y;

            // 边界检查
            int src_x_next = std::min(src_x + 1, src_width - 1);
            int src_y_next = std::min(src_y + 1, src_height - 1);

            for (int c = 0; c < channels; c++)
            {
                // 四个相邻像素
                unsigned char A = src[(src_y * src_width + src_x) * channels + c];
                unsigned char B = src[(src_y * src_width + src_x_next) * channels + c];
                unsigned char C = src[(src_y_next * src_width + src_x) * channels + c];
                unsigned char D = src[(src_y_next * src_width + src_x_next) * channels + c];

                // 双线性插值
                float pixel = A * (1 - x_diff) * (1 - y_diff) + B * x_diff * (1 - y_diff) + C * (1 - x_diff) * y_diff +
                              D * x_diff * y_diff;

                dst[(dst_y * dst_width + dst_x) * channels + c] = static_cast<unsigned char>(pixel);
            }
        }
    }

    return true;
}

// 灰度图像裁剪和缩放
bool cropAndScaleGRAY(const unsigned char* src, int src_width, int src_height, int crop_x, int crop_y, int crop_width,
                      int crop_height, unsigned char* dst, int dst_width, int /*dst_height*/, int dst_box_x,
                      int dst_box_y, int dst_box_width, int dst_box_height)
{
    if (src == nullptr || dst == nullptr)
    {
        std::cerr << "Null pointer provided to cropAndScaleGRAY" << std::endl;
        return false;
    }

    float x_ratio = static_cast<float>(crop_width) / static_cast<float>(dst_box_width);
    float y_ratio = static_cast<float>(crop_height) / static_cast<float>(dst_box_height);

    // 双线性插值缩放
    for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++)
    {
        for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++)
        {
            int dst_x_offset = dst_x - dst_box_x;
            int dst_y_offset = dst_y - dst_box_y;

            float src_x_f = dst_x_offset * x_ratio + crop_x;
            float src_y_f = dst_y_offset * y_ratio + crop_y;

            int src_x = static_cast<int>(src_x_f);
            int src_y = static_cast<int>(src_y_f);

            float x_diff = src_x_f - src_x;
            float y_diff = src_y_f - src_y;

            // 边界检查
            int src_x_next = std::min(src_x + 1, src_width - 1);
            int src_y_next = std::min(src_y + 1, src_height - 1);

            // 四个相邻像素
            unsigned char A = src[src_y * src_width + src_x];
            unsigned char B = src[src_y * src_width + src_x_next];
            unsigned char C = src[src_y_next * src_width + src_x];
            unsigned char D = src[src_y_next * src_width + src_x_next];

            // 双线性插值
            float pixel = A * (1 - x_diff) * (1 - y_diff) + B * x_diff * (1 - y_diff) + C * (1 - x_diff) * y_diff +
                          D * x_diff * y_diff;

            dst[dst_y * dst_width + dst_x] = static_cast<unsigned char>(pixel);
        }
    }

    return true;
}

// YUV420SP图像裁剪和缩放
bool cropAndScaleYUV420SP(const unsigned char* src, int src_width, int src_height, int crop_x, int crop_y,
                          int crop_width, int crop_height, unsigned char* dst, int dst_width, int dst_height,
                          int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height)
{
    if (src == nullptr || dst == nullptr)
    {
        std::cerr << "Null pointer provided to cropAndScaleYUV420SP" << std::endl;
        return false;
    }

    const unsigned char* src_y = src;
    const unsigned char* src_uv = src + src_width * src_height;

    unsigned char* dst_y = dst;
    unsigned char* dst_uv = dst + dst_width * dst_height;

    // 处理Y分量
    if (!cropAndScaleGRAY(src_y, src_width, src_height, crop_x, crop_y, crop_width, crop_height, dst_y, dst_width,
                          dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height))
    {
        return false;
    }

    // 处理UV分量 (半分辨率)
    return cropAndScaleRGB(src_uv, src_width / 2, src_height / 2, crop_x / 2, crop_y / 2, crop_width / 2,
                           crop_height / 2, dst_uv, dst_width / 2, dst_height / 2, dst_box_x / 2, dst_box_y / 2,
                           dst_box_width / 2, dst_box_height / 2);
}

// 通用图像转换和缩放函数
bool convertImageGeneric(const image_buffer_t& src_image, image_buffer_t& dst_image, int src_box_x, int src_box_y,
                         int src_box_w, int src_box_h, int dst_box_x, int dst_box_y, int dst_box_w, int dst_box_h,
                         unsigned char bg_color)
{
    if (src_image.virt_addr == nullptr || dst_image.virt_addr == nullptr)
    {
        std::cerr << "Null image buffer provided" << std::endl;
        return false;
    }

    if (src_image.format != dst_image.format)
    {
        std::cerr << "Source and destination formats must match" << std::endl;
        return false;
    }

    // 填充背景色
    if (dst_box_w != dst_image.width || dst_box_h != dst_image.height)
    {
        memset(dst_image.virt_addr, bg_color, dst_image.size);
    }

    // 根据格式选择处理函数
    switch (src_image.format)
    {
        case ImageFormat::RGB888:
            return cropAndScaleRGB(src_image.virt_addr, src_image.width, src_image.height, src_box_x, src_box_y,
                                   src_box_w, src_box_h, dst_image.virt_addr, dst_image.width, dst_image.height,
                                   dst_box_x, dst_box_y, dst_box_w, dst_box_h);

        case ImageFormat::GRAY8:
            return cropAndScaleGRAY(src_image.virt_addr, src_image.width, src_image.height, src_box_x, src_box_y,
                                    src_box_w, src_box_h, dst_image.virt_addr, dst_image.width, dst_image.height,
                                    dst_box_x, dst_box_y, dst_box_w, dst_box_h);

        case ImageFormat::YUV420SP_NV12:
        case ImageFormat::YUV420SP_NV21:
            return cropAndScaleYUV420SP(src_image.virt_addr, src_image.width, src_image.height, src_box_x, src_box_y,
                                        src_box_w, src_box_h, dst_image.virt_addr, dst_image.width, dst_image.height,
                                        dst_box_x, dst_box_y, dst_box_w, dst_box_h);

        case ImageFormat::RGBA8888:
            // RGBA按4通道RGB处理 (需要修改cropAndScaleRGB支持4通道)
            std::cerr << "RGBA8888 format not fully implemented yet" << std::endl;
            return false;

        default:
            std::cerr << "Unsupported image format: " << static_cast<int>(src_image.format) << std::endl;
            return false;
    }
}

// Letterbox预处理 (保持长宽比)
bool letterboxResize(const image_buffer_t& src_image, image_buffer_t& dst_image, int target_width, int target_height,
                     unsigned char bg_color)
{
    LetterboxParams params;
    return letterboxResizeWithParams(src_image, dst_image, target_width, target_height, params, bg_color);
}

// Letterbox预处理 (带参数返回，用于后处理坐标转换)
bool letterboxResizeWithParams(const image_buffer_t& src_image, image_buffer_t& dst_image, int target_width,
                               int target_height, LetterboxParams& letterbox_params, unsigned char bg_color)
{
    // 确保目标图像缓冲区已创建
    if (dst_image.virt_addr == nullptr)
    {
        dst_image = createImageBuffer(target_width, target_height, src_image.format);
        if (dst_image.virt_addr == nullptr)
        {
            return false;
        }
    }

    // 计算缩放比例，保持长宽比（与yolov3-rknn保持一致）
    float scale_x = static_cast<float>(target_width) / src_image.width;
    float scale_y = static_cast<float>(target_height) / src_image.height;
    letterbox_params.scale = std::min(scale_x, scale_y);

    int scaled_width = static_cast<int>(src_image.width * letterbox_params.scale);
    int scaled_height = static_cast<int>(src_image.height * letterbox_params.scale);

    // 计算居中位置
    letterbox_params.x_pad = (target_width - scaled_width) / 2;
    letterbox_params.y_pad = (target_height - scaled_height) / 2;

    return convertImageGeneric(src_image, dst_image, 0, 0, src_image.width, src_image.height,  // 使用整个源图像
                               letterbox_params.x_pad, letterbox_params.y_pad, scaled_width, scaled_height,  // 目标区域
                               bg_color);
}

// 标准缩放 (拉伸到目标尺寸)
bool standardResize(const image_buffer_t& src_image, image_buffer_t& dst_image, int target_width, int target_height)
{
    // 确保目标图像缓冲区已创建
    if (dst_image.virt_addr == nullptr)
    {
        dst_image = createImageBuffer(target_width, target_height, src_image.format);
        if (dst_image.virt_addr == nullptr)
        {
            return false;
        }
    }

    return convertImageGeneric(src_image, dst_image, 0, 0, src_image.width, src_image.height,  // 使用整个源图像
                               0, 0, target_width, target_height,                              // 目标整个图像
                               0);
}

// 通用转换接口
bool convertImage(const image_buffer_t& src_image, image_buffer_t& dst_image, bool letterbox, unsigned char bg_color)
{
    if (letterbox)
    {
        return letterboxResize(src_image, dst_image, dst_image.width, dst_image.height, bg_color);
    }
    else
    {
        return standardResize(src_image, dst_image, dst_image.width, dst_image.height);
    }
}

// 图像裁剪
bool cropImage(const image_buffer_t& src_image, image_buffer_t& dst_image, int crop_x, int crop_y, int crop_width,
               int crop_height)
{
    return convertImageGeneric(src_image, dst_image, crop_x, crop_y, crop_width, crop_height,  // 源区域
                               0, 0, dst_image.width, dst_image.height,                        // 目标整个图像
                               0);
}

// 图像归一化
bool normalizeImage(const image_buffer_t& image, float* normalized_data, const float mean[3], const float std[3],
                    bool is_nchw)
{
    if (image.virt_addr == nullptr || normalized_data == nullptr)
    {
        std::cerr << "Null pointer provided to normalizeImage" << std::endl;
        return false;
    }

    if (image.format != ImageFormat::RGB888)
    {
        std::cerr << "Only RGB888 format supported for normalization" << std::endl;
        return false;
    }

    // 默认归一化参数
    float default_mean[3] = {0.0f, 0.0f, 0.0f};
    float default_std[3] = {255.0f, 255.0f, 255.0f};

    const float* use_mean = mean ? mean : default_mean;
    const float* use_std = std ? std : default_std;

    int width = image.width;
    int height = image.height;
    int channels = 3;

    if (is_nchw)
    {
        // NCHW format: [C, H, W]
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int src_idx = (h * width + w) * channels + c;
                    int dst_idx = c * height * width + h * width + w;

                    float pixel = static_cast<float>(image.virt_addr[src_idx]);
                    normalized_data[dst_idx] = (pixel - use_mean[c]) / use_std[c];
                }
            }
        }
    }
    else
    {
        // NHWC format: [H, W, C]
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int c = 0; c < channels; c++)
                {
                    int idx = (h * width + w) * channels + c;
                    float pixel = static_cast<float>(image.virt_addr[idx]);
                    normalized_data[idx] = (pixel - use_mean[c]) / use_std[c];
                }
            }
        }
    }

    return true;
}

}  // namespace utils
}  // namespace rknn_cpp
