#pragma once
#include <memory>
#include <vector>
#include <string>
#include <any>

namespace rknn_cpp
{

// 模型任务类型枚举
enum class ModelTask
{
    OBJECT_DETECTION,
    CLASSIFICATION,
    UNKNOWN
};

// 图像格式枚举
enum class ImageFormat
{
    GRAY8,
    RGB888,
    RGBA8888,
    YUV420SP_NV21,
    YUV420SP_NV12
};

// 图像缓冲区结构
struct image_buffer_t
{
    int width;
    int height;
    int width_stride;
    int height_stride;
    ImageFormat format;
    unsigned char* virt_addr;
    int size;
    int fd;
};

// 预处理配置基类
struct PreprocessConfig
{
    int target_width;
    int target_height;
    int target_channels;
    virtual ~PreprocessConfig() = default;
};

// Letterbox预处理配置
struct LetterboxConfig : public PreprocessConfig
{
    int bg_color = 114;
    bool keep_aspect_ratio = true;
};

// 标准Resize配置
struct ResizeConfig : public PreprocessConfig
{
    bool keep_aspect_ratio = false;
};

// ===== 推理结果类型定义 =====

// 检测结果
struct DetectionResult
{
    float x, y, width, height;  // 边界框坐标和尺寸
    float confidence;           // 置信度
    int class_id;               // 类别ID
    std::string class_name;     // 类别名称
};

// 分类结果
struct ClassificationResult
{
    int class_id;            // 类别ID
    std::string class_name;  // 类别名称
    float confidence;        // 置信度
};

// 推理结果的集合类型
using DetectionResults = std::vector<DetectionResult>;
using ClassificationResults = std::vector<ClassificationResult>;

// 通用推理结果
struct InferenceResult
{
    ModelTask task_type;
    std::any result_data;  // 可以是DetectionResults或ClassificationResults

    // 便利方法
    DetectionResults getDetections() const
    {
        if (task_type == ModelTask::OBJECT_DETECTION)
        {
            try
            {
                return std::any_cast<DetectionResults>(result_data);
            }
            catch (const std::bad_any_cast& e)
            {
                // 处理类型转换失败的情况
                return {};
            }
        }
        return {};
    }

    ClassificationResults getClassifications() const
    {
        if (task_type == ModelTask::CLASSIFICATION)
        {
            try
            {
                return std::any_cast<ClassificationResults>(result_data);
            }
            catch (const std::bad_any_cast& e)
            {
                // 处理类型转换失败的情况
                return {};
            }
        }
        return {};
    }
};

}  // namespace rknn_cpp