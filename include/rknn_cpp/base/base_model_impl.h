#pragma once
#include "rknn_cpp/imodel.h"
#include "rknn_api.h"
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace rknn_cpp
{

class BaseModelImpl : public IModel
{
   public:
    BaseModelImpl();
    virtual ~BaseModelImpl();

    // 实现IModel接口
    bool initialize(const ModelConfig& config = {}) override final;
    InferenceResult predict(const image_buffer_t& image) override final;  // 更新返回类型
    InferenceResult predict(const cv::Mat& image);                        // 新增cv::Mat重载
    void release() override;
    bool isInitialized() const override;
    int getModelWidth() const override;
    int getModelHeight() const override;
    int getModelChannels() const override;
    int getOriginalWidth() const { return original_width_; }
    int getOriginalHeight() const { return original_height_; }

   protected:
    // 子类需要实现的抽象方法
    virtual bool setupModel(const ModelConfig& config) = 0;
    virtual bool preprocessImage(const image_buffer_t& src_img, image_buffer_t& dst_img) = 0;
    virtual bool preprocessImage(const cv::Mat& src_img, cv::Mat& dst_img) = 0;              // 新增cv::Mat重载
    virtual InferenceResult postprocessOutputs(rknn_output* outputs, int output_count) = 0;  // 更新返回类型

    // 为子类提供的工具方法
    bool loadRKNNModel(const std::string& model_path);
    bool runRKNNInference(const image_buffer_t& input_img);
    bool runRKNNInference(const cv::Mat& input_img);  // 新增cv::Mat重载
    void dumpTensorAttrs() const;

    // 为子类提供的便利方法 - 创建结果对象
    InferenceResult createDetectionResult(const DetectionResults& detections) const;
    InferenceResult createClassificationResult(const ClassificationResults& classifications) const;
    InferenceResult createEmptyResult() const;

    // 为子类提供的图像处理帮助方法
    image_buffer_t createModelSizedBuffer() const;
    bool standardPreprocess(const image_buffer_t& src_img, image_buffer_t& dst_img) const;
    bool standardPreprocess(const cv::Mat& src_img, cv::Mat& dst_img) const;  // 新增cv::Mat重载
    bool letterboxPreprocess(const image_buffer_t& src_img, image_buffer_t& dst_img,
                             unsigned char bg_color = 114) const;
    bool letterboxPreprocess(const cv::Mat& src_img, cv::Mat& dst_img,
                             unsigned char bg_color = 114) const;  // 新增cv::Mat重载
    void freeImageBuffer(image_buffer_t& image) const;

    // 为子类提供的模型属性访问
    bool isQuantized() const { return is_quant_; }
    const std::vector<rknn_tensor_attr>& getInputAttrs() const { return input_attrs_; }
    const std::vector<rknn_tensor_attr>& getOutputAttrs() const { return output_attrs_; }
    rknn_context getRKNNContext() const { return rknn_ctx_; }

   private:
    rknn_context rknn_ctx_;
    rknn_input_output_num io_num_;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    int model_width_;
    int model_height_;
    int model_channels_;
    int original_width_;   // 原始输入图像宽度
    int original_height_;  // 原始输入图像高度
    bool initialized_;
    bool is_quant_;

    // 输出缓冲区
    std::vector<rknn_output> outputs_;

    // 预处理缓冲区
    image_buffer_t preprocess_buffer_;
};

}  // namespace rknn_cpp