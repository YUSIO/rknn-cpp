#pragma once
#include "rknn_cpp/types.h"
#include <string>

namespace rknn_cpp
{
namespace utils
{

/**
 * @brief 计算图像缓冲区大小
 * @param image 图像缓冲区结构
 * @return 字节数大小
 */
int getImageSize(const image_buffer_t& image);

/**
 * @brief 读取图像文件 (支持 png/jpeg/bmp)
 * @param path 图像路径
 * @param image 输出图像缓冲区
 * @return true: 成功; false: 失败
 */
bool readImage(const std::string& path, image_buffer_t& image);

/**
 * @brief 释放图像缓冲区内存
 * @param image 图像缓冲区
 */
void freeImage(image_buffer_t& image);

/**
 * @brief 图像格式转换和尺寸调整
 * @param src_image 源图像
 * @param dst_image 目标图像 (需要预先设置宽高和格式)
 * @param letterbox 是否使用letterbox填充 (保持长宽比)
 * @param bg_color 填充颜色 (letterbox=true时使用)
 * @return true: 成功; false: 失败
 */
bool convertImage(const image_buffer_t& src_image, image_buffer_t& dst_image, bool letterbox = true,
                  unsigned char bg_color = 114);

/**
 * @brief Letterbox预处理参数
 * 与yolov3-rknn项目保持一致的数据结构
 */
struct LetterboxParams
{
    int x_pad;
    int y_pad;
    float scale;
};

/**
 * @brief Letterbox预处理 (保持长宽比，填充灰色)
 * @param src_image 源图像
 * @param dst_image 目标图像
 * @param target_width 目标宽度
 * @param target_height 目标高度
 * @param bg_color 填充颜色
 * @return true: 成功; false: 失败
 */
bool letterboxResize(const image_buffer_t& src_image, image_buffer_t& dst_image, int target_width, int target_height,
                     unsigned char bg_color = 114);

/**
 * @brief Letterbox预处理 (带参数返回，用于后处理坐标转换)
 * @param src_image 源图像
 * @param dst_image 目标图像
 * @param target_width 目标宽度
 * @param target_height 目标高度
 * @param letterbox_params 返回的letterbox参数
 * @param bg_color 填充颜色
 * @return true: 成功; false: 失败
 */
bool letterboxResizeWithParams(const image_buffer_t& src_image, image_buffer_t& dst_image, int target_width,
                               int target_height, LetterboxParams& letterbox_params, unsigned char bg_color = 114);

/**
 * @brief 标准缩放 (拉伸到目标尺寸，不保持长宽比)
 * @param src_image 源图像
 * @param dst_image 目标图像
 * @param target_width 目标宽度
 * @param target_height 目标高度
 * @return true: 成功; false: 失败
 */
bool standardResize(const image_buffer_t& src_image, image_buffer_t& dst_image, int target_width, int target_height);

/**
 * @brief 图像裁剪
 * @param src_image 源图像
 * @param dst_image 目标图像
 * @param crop_x 裁剪起始x坐标
 * @param crop_y 裁剪起始y坐标
 * @param crop_width 裁剪宽度
 * @param crop_height 裁剪高度
 * @return true: 成功; false: 失败
 */
bool cropImage(const image_buffer_t& src_image, image_buffer_t& dst_image, int crop_x, int crop_y, int crop_width,
               int crop_height);

/**
 * @brief 图像归一化 (转换为浮点数并归一化到 [0,1] 或 [-1,1])
 * @param image 输入图像 (UINT8格式)
 * @param normalized_data 输出归一化数据 (FP32格式)
 * @param mean RGB均值 [R_mean, G_mean, B_mean]
 * @param std RGB标准差 [R_std, G_std, B_std]
 * @param is_nchw 是否转换为NCHW格式 (true: NCHW, false: NHWC)
 * @return true: 成功; false: 失败
 */
bool normalizeImage(const image_buffer_t& image, float* normalized_data, const float mean[3] = nullptr,
                    const float std[3] = nullptr, bool is_nchw = false);

/**
 * @brief 创建图像缓冲区
 * @param width 宽度
 * @param height 高度
 * @param format 图像格式
 * @return 创建的图像缓冲区
 */
image_buffer_t createImageBuffer(int width, int height, ImageFormat format);

/**
 * @brief 复制图像缓冲区
 * @param src 源图像
 * @return 复制的图像缓冲区
 */
image_buffer_t cloneImageBuffer(const image_buffer_t& src);

/**
 * @brief 打印图像信息 (调试用)
 * @param image 图像缓冲区
 * @param name 图像名称
 */
void printImageInfo(const image_buffer_t& image, const std::string& name = "Image");

// =====  内部辅助函数 =====

/**
 * @brief RGB图像裁剪和缩放 (双线性插值)
 * @param src 源图像数据
 * @param src_width 源图像宽度
 * @param src_height 源图像高度
 * @param crop_x 裁剪起始x坐标
 * @param crop_y 裁剪起始y坐标
 * @param crop_width 裁剪宽度
 * @param crop_height 裁剪高度
 * @param dst 目标图像数据
 * @param dst_width 目标图像宽度
 * @param dst_height 目标图像高度
 * @param dst_box_x 目标区域起始x坐标
 * @param dst_box_y 目标区域起始y坐标
 * @param dst_box_width 目标区域宽度
 * @param dst_box_height 目标区域高度
 * @return true: 成功; false: 失败
 */
bool cropAndScaleRGB(const unsigned char* src, int src_width, int src_height, int crop_x, int crop_y, int crop_width,
                     int crop_height, unsigned char* dst, int dst_width, int dst_height, int dst_box_x, int dst_box_y,
                     int dst_box_width, int dst_box_height);

/**
 * @brief 灰度图像裁剪和缩放 (双线性插值)
 */
bool cropAndScaleGRAY(const unsigned char* src, int src_width, int src_height, int crop_x, int crop_y, int crop_width,
                      int crop_height, unsigned char* dst, int dst_width, int dst_height, int dst_box_x, int dst_box_y,
                      int dst_box_width, int dst_box_height);

/**
 * @brief YUV420SP图像裁剪和缩放
 */
bool cropAndScaleYUV420SP(const unsigned char* src, int src_width, int src_height, int crop_x, int crop_y,
                          int crop_width, int crop_height, unsigned char* dst, int dst_width, int dst_height,
                          int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height);

}  // namespace utils
}  // namespace rknn_cpp
