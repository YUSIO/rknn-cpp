#pragma once
#include "rknn_cpp/base/base_model_impl.h"
#include <vector>

namespace rknn_cpp
{
struct LetterboxParams
{
    int x_pad;
    int y_pad;
    float scale;
};
/**
 * @brief Yolov3检测模型实现
 * 基于BaseModelImpl，提供Yolov3模型的检测功能
 */
class Yolov3Model : public BaseModelImpl
{
   public:
    Yolov3Model();
    virtual ~Yolov3Model() = default;

    // 实现IModel接口
    ModelTask getTaskType() const override;
    std::string getModelName() const override;

   protected:
    // 实现BaseModelImpl的抽象方法
    bool setupModel(const ModelConfig& config) override;
    bool preprocessImage(const cv::Mat& src_img, cv::Mat& dst_img) override;  // 新增cv::Mat重载
    InferenceResult postprocessOutputs(rknn_output* outputs, int output_count) override;

   private:
    // 成员变量
    std::vector<std::string> class_names_;
    bool class_names_loaded_;
    LetterboxParams letterbox_params_;  // 保存letterbox预处理参数
    struct YoloLayer
    {
        int grid_h;
        int grid_w;
        int stride;
        std::vector<int> anchors;
    };
    // 类别名称相关
    bool loadClassNames(const std::string& filepath);
    std::string getClassName(int class_id) const;
    // 工具函数
    float sigmoid(float x) const;
    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) const;
    int processYoloLayer(void* input, bool is_quantized, const YoloLayer& layer, std::vector<float>& boxes,
                         std::vector<float>& objProbs, std::vector<int>& classId, float threshold, int32_t zp = 0,
                         float scale = 0) const;

    std::vector<int> applyNMS(const std::vector<float>& boxes, const std::vector<float>& scores,
                              const std::vector<int>& classIds, float nms_threshold = 0.45f) const;

    int nmsForClass(const std::vector<float>& boxes, const std::vector<int>& classIds, std::vector<int>& order,
                    int filterId, float threshold) const;

    float calculateIoU(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                       float ymax1) const;

    // Letterbox坐标转换
    void convertLetterboxToOriginal(DetectionResults& detections, int orig_width, int orig_height) const;
};
}  // namespace rknn_cpp