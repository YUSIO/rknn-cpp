#pragma once
#include "rknn_cpp/base/base_model_impl.h"
#include <vector>

namespace rknn_cpp
{

/**
 * @brief ResNet分类模型实现
 * 基于BaseModelImpl，提供ResNet系列模型的分类功能
 */
class ResNetModel : public BaseModelImpl
{
   public:
    ResNetModel();
    virtual ~ResNetModel() = default;

    // 实现IModel接口
    ModelTask getTaskType() const override;
    std::string getModelName() const override;

   protected:
    // 实现BaseModelImpl的抽象方法
    bool setupModel(const ModelConfig& config) override;
    bool preprocessImage(const image_buffer_t& src_img, image_buffer_t& dst_img) override;
    bool preprocessImage(const cv::Mat& src_img, cv::Mat& dst_img) override;  // 新增cv::Mat重载
    InferenceResult postprocessOutputs(rknn_output* outputs, int output_count) override;

   private:
    // ResNet特定的后处理方法
    void applySoftmax(float* data, int size);
    ClassificationResults getTopK(const float* data, int size, int k = 5);

    // 类别名称相关
    bool loadClassNames(const std::string& filepath);
    std::string getClassName(int class_id) const;

    // 成员变量
    std::vector<std::string> class_names_;
    bool class_names_loaded_;

    // 反量化
    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) const;
};
}  // namespace rknn_cpp