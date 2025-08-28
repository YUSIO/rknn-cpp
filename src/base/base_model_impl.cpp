#include "rknn_cpp/base/base_model_impl.h"
#include "rknn_cpp/utils/image_utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <chrono>

namespace rknn_cpp
{

BaseModelImpl::BaseModelImpl()
    : rknn_ctx_(0),
      model_width_(0),
      model_height_(0),
      model_channels_(0),
      initialized_(false),
      is_quant_(false),
      preprocess_buffer_{}
{
    memset(&io_num_, 0, sizeof(io_num_));
}

BaseModelImpl::~BaseModelImpl()
{
    release();
}

bool BaseModelImpl::initialize(const ModelConfig& config)
{
    if (initialized_)
    {
        std::cout << "\n[MODEL] Already initialized" << std::endl;
        return true;
    }

    // 1. 加载RKNN模型
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "                  MODEL INITIALIZATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    auto model_path_it = config.find("model_path");
    if (model_path_it == config.end() || model_path_it->second.empty())
    {
        std::cerr << "Model path not specified in config" << std::endl;
        return false;
    }
    std::string model_path = model_path_it->second;
    std::cout << "[LOAD] Loading model file: " << model_path << std::endl;
    if (!loadRKNNModel(model_path))
    {
        std::cerr << "Failed to load RKNN model: " << model_path << std::endl;
        return false;
    }

    // 2. 获取模型输入输出信息
    int ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_query RKNN_QUERY_IN_OUT_NUM failed! ret=" << ret << std::endl;
        return false;
    }
    std::cout << "[INFO] Model I/O Configuration" << std::endl;
    std::cout << "       Input Tensors : " << io_num_.n_input << std::endl;
    std::cout << "       Output Tensors: " << io_num_.n_output << std::endl;

    // 3. 获取输入属性
    input_attrs_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; i++)
    {
        input_attrs_[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::cerr << "rknn_query RKNN_QUERY_INPUT_ATTR failed! ret=" << ret << std::endl;
            return false;
        }
    }

    // 4. 获取输出属性
    output_attrs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; i++)
    {
        output_attrs_[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::cerr << "rknn_query RKNN_QUERY_OUTPUT_ATTR failed! ret=" << ret << std::endl;
            return false;
        }
    }
    if (io_num_.n_output > 0)
    {
        const auto& out_attr = output_attrs_[0];
        if (out_attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && out_attr.type != RKNN_TENSOR_FLOAT16)
        {
            is_quant_ = true;
        }
        else
        {
            is_quant_ = false;
        }
    }
    // 5. 提取模型输入尺寸信息（假设第一个输入是图像）
    if (io_num_.n_input > 0)
    {
        auto& input_attr = input_attrs_[0];
        if (input_attr.n_dims == 4)
        {  // NHWC or NCHW
            if (input_attr.fmt == RKNN_TENSOR_NHWC)
            {
                model_height_ = input_attr.dims[1];
                model_width_ = input_attr.dims[2];
                model_channels_ = input_attr.dims[3];
            }
            else if (input_attr.fmt == RKNN_TENSOR_NCHW)
            {
                model_channels_ = input_attr.dims[1];
                model_height_ = input_attr.dims[2];
                model_width_ = input_attr.dims[3];
            }
        }
    }

    // 6. 打印张量信息
    dumpTensorAttrs();

    // 7. 初始化输出缓冲区
    outputs_.resize(io_num_.n_output);
    memset(outputs_.data(), 0, outputs_.size() * sizeof(rknn_output));

    // 8. 分配预处理缓冲区（按模型输入尺寸）
    preprocess_buffer_ = utils::createImageBuffer(model_width_, model_height_, ImageFormat::RGB888);
    if (preprocess_buffer_.virt_addr == nullptr)
    {
        std::cerr << "Failed to allocate preprocess buffer" << std::endl;
        return false;
    }

    // 9. 调用子类的模型设置
    if (!setupModel(config))
    {
        std::cerr << "setupModel failed!" << std::endl;
        return false;
    }

    initialized_ = true;
    std::cout << "\n[SUCCESS] Model initialization completed" << std::endl;
    std::cout << "[CONFIG] Input Dimensions: " << model_width_ << " x " << model_height_ << " x " << model_channels_
              << std::endl;
    std::cout << "[CONFIG] Quantization   : " << (is_quant_ ? "Enabled" : "Disabled") << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    return true;
}

InferenceResult BaseModelImpl::predict(const image_buffer_t& image)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    if (!initialized_)
    {
        std::cerr << "Model not initialized!" << std::endl;
        return createEmptyResult();
    }

    // 1. 预处理图像到复用缓冲区
    if (preprocess_buffer_.virt_addr)
    {
        memset(preprocess_buffer_.virt_addr, 0, preprocess_buffer_.size);
    }

    if (!preprocessImage(image, preprocess_buffer_))
    {
        std::cerr << "Image preprocessing failed!" << std::endl;
        return createEmptyResult();
    }
    std::chrono::steady_clock::time_point point1 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> preprocess_duration = point1 - start;
    std::cout << "[INFO] Image preprocessing time: " << preprocess_duration.count() << " ms" << std::endl;

    // 2. 执行推理
    if (!runRKNNInference(preprocess_buffer_))
    {
        std::cerr << "RKNN inference failed!" << std::endl;
        return createEmptyResult();
    }
    std::chrono::steady_clock::time_point point2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> inference_duration = point2 - point1;
    std::cout << "[INFO] RKNN inference time: " << inference_duration.count() << " ms" << std::endl;

    // 3. 后处理
    InferenceResult result = postprocessOutputs(outputs_.data(), outputs_.size());
    std::chrono::steady_clock::time_point point3 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> postprocess_duration = point3 - point2;
    std::cout << "[INFO] Postprocess time: " << postprocess_duration.count() << " ms" << std::endl;
    std::cout << "[INFO] Total inference time: "
              << (preprocess_duration + inference_duration + postprocess_duration).count() << " ms" << std::endl;
    // 4. 释放输出资源
    rknn_outputs_release(rknn_ctx_, io_num_.n_output, outputs_.data());

    // 5. 返回推理结果（预处理缓冲区在成员中复用，不再释放）
    return result;
}

void BaseModelImpl::release()
{
    if (!initialized_)
    {
        return;  // 已经释放过了，直接返回
    }

    if (rknn_ctx_ != 0)
    {
        rknn_destroy(rknn_ctx_);
        rknn_ctx_ = 0;
    }

    outputs_.clear();

    // 释放预处理缓冲区
    utils::freeImage(preprocess_buffer_);

    input_attrs_.clear();
    output_attrs_.clear();

    initialized_ = false;
    std::cout << "\n[RELEASE] Model resources freed" << std::endl;
}

bool BaseModelImpl::isInitialized() const
{
    return initialized_;
}

int BaseModelImpl::getModelWidth() const
{
    return model_width_;
}

int BaseModelImpl::getModelHeight() const
{
    return model_height_;
}

int BaseModelImpl::getModelChannels() const
{
    return model_channels_;
}

// ===== Protected 工具方法实现 =====

bool BaseModelImpl::loadRKNNModel(const std::string& model_path)
{
    // 1. 读取模型文件
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Cannot open model file: " << model_path << std::endl;
        return false;
    }

    size_t model_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> model_data(model_size);
    if (!file.read(model_data.data(), model_size))
    {
        std::cerr << "Failed to read model file" << std::endl;
        return false;
    }
    file.close();

    std::cout << "[INFO] Model file size: " << model_size << " bytes" << std::endl;

    // 2. 初始化RKNN
    int ret = rknn_init(&rknn_ctx_, model_data.data(), model_size, 0, nullptr);
    if (ret < 0)
    {
        std::cerr << "rknn_init failed! ret=" << ret << std::endl;
        return false;
    }

    return true;
}

bool BaseModelImpl::runRKNNInference(const image_buffer_t& input_img)
{
    // 1. 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = input_img.virt_addr;
    inputs[0].size = input_img.size;
    inputs[0].pass_through = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(rknn_ctx_, io_num_.n_input, inputs);
    if (ret < 0)
    {
        std::cerr << "rknn_inputs_set failed! ret=" << ret << std::endl;
        return false;
    }

    // 2. 执行推理
    ret = rknn_run(rknn_ctx_, nullptr);
    if (ret < 0)
    {
        std::cerr << "rknn_run failed! ret=" << ret << std::endl;
        return false;
    }

    // 3. 获取输出 - 设置want_float以获取浮点数据
    for (uint32_t i = 0; i < io_num_.n_output; i++)
    {
        outputs_[i].want_float = (!is_quant_);
    }

    ret = rknn_outputs_get(rknn_ctx_, io_num_.n_output, outputs_.data(), nullptr);
    if (ret < 0)
    {
        std::cerr << "rknn_outputs_get failed! ret=" << ret << std::endl;
        return false;
    }

    return true;
}

void BaseModelImpl::dumpTensorAttrs() const
{
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                           MODEL TENSOR INFORMATION" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // 输入张量信息
    std::cout << "\n[INPUT TENSORS]" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (size_t i = 0; i < input_attrs_.size(); i++)
    {
        const auto& attr = input_attrs_[i];
        std::cout << "Input[" << i << "]: " << (attr.name ? attr.name : "unnamed") << std::endl;
        std::cout << "  Index      : " << attr.index << std::endl;
        std::cout << "  Dimensions : ";
        for (uint32_t j = 0; j < attr.n_dims; j++)
        {
            std::cout << attr.dims[j];
            if (j < attr.n_dims - 1) std::cout << " x ";
        }
        std::cout << " (" << attr.n_dims << "D)" << std::endl;
        std::cout << "  Elements   : " << attr.n_elems << std::endl;
        std::cout << "  Size       : " << attr.size << " bytes" << std::endl;
        std::cout << "  Format     : " << get_format_string(attr.fmt) << std::endl;
        std::cout << "  Type       : " << get_type_string(attr.type) << std::endl;
        std::cout << "  Quant Type : " << get_qnt_type_string(attr.qnt_type) << std::endl;

        if (i < input_attrs_.size() - 1)
        {
            std::cout << std::string(30, '.') << std::endl;
        }
    }

    // 输出张量信息
    std::cout << "\n[OUTPUT TENSORS]" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (size_t i = 0; i < output_attrs_.size(); i++)
    {
        const auto& attr = output_attrs_[i];
        std::cout << "Output[" << i << "]: " << (attr.name ? attr.name : "unnamed") << std::endl;
        std::cout << "  Index      : " << attr.index << std::endl;
        std::cout << "  Dimensions : ";
        for (uint32_t j = 0; j < attr.n_dims; j++)
        {
            std::cout << attr.dims[j];
            if (j < attr.n_dims - 1) std::cout << " x ";
        }
        std::cout << " (" << attr.n_dims << "D)" << std::endl;
        std::cout << "  Elements   : " << attr.n_elems << std::endl;
        std::cout << "  Size       : " << attr.size << " bytes" << std::endl;
        std::cout << "  Format     : " << get_format_string(attr.fmt) << std::endl;
        std::cout << "  Type       : " << get_type_string(attr.type) << std::endl;
        std::cout << "  Quant Type : " << get_qnt_type_string(attr.qnt_type) << std::endl;

        // 如果是量化张量，显示量化参数
        if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
        {
            std::cout << "  Zero Point : " << attr.zp << std::endl;
            std::cout << "  Scale      : " << std::fixed << std::setprecision(6) << attr.scale << std::endl;
        }

        if (i < output_attrs_.size() - 1)
        {
            std::cout << std::string(30, '.') << std::endl;
        }
    }

    std::cout << std::string(80, '=') << std::endl;
}

// ===== 便利方法实现 =====

InferenceResult BaseModelImpl::createDetectionResult(const DetectionResults& detections) const
{
    InferenceResult result;
    result.task_type = ModelTask::OBJECT_DETECTION;
    result.result_data = detections;
    return result;
}

InferenceResult BaseModelImpl::createClassificationResult(const ClassificationResults& classifications) const
{
    InferenceResult result;
    result.task_type = ModelTask::CLASSIFICATION;
    result.result_data = classifications;
    return result;
}

InferenceResult BaseModelImpl::createEmptyResult() const
{
    InferenceResult result;
    result.task_type = getTaskType();

    switch (getTaskType())
    {
        case ModelTask::OBJECT_DETECTION:
            result.result_data = DetectionResults{};
            break;
        case ModelTask::CLASSIFICATION:
            result.result_data = ClassificationResults{};
            break;
        default:
            result.result_data = DetectionResults{};
            break;
    }

    return result;
}

// ===== 图像处理帮助方法 =====

image_buffer_t BaseModelImpl::createModelSizedBuffer() const
{
    return preprocess_buffer_;
}

bool BaseModelImpl::standardPreprocess(const image_buffer_t& src_img, image_buffer_t& dst_img) const
{
    dst_img = createModelSizedBuffer();
    if (dst_img.virt_addr == nullptr)
    {
        return false;
    }

    // 使用标准缩放 (拉伸到目标尺寸，不保持长宽比)
    if (!utils::standardResize(src_img, dst_img, model_width_, model_height_))
    {
        return false;
    }

    return true;
}

bool BaseModelImpl::letterboxPreprocess(const image_buffer_t& src_img, image_buffer_t& dst_img,
                                        unsigned char bg_color) const
{
    dst_img = createModelSizedBuffer();
    if (dst_img.virt_addr == nullptr)
    {
        return false;
    }

    // 使用letterbox (保持长宽比，填充背景色)
    if (!utils::letterboxResize(src_img, dst_img, model_width_, model_height_, bg_color))
    {
        return false;
    }

    return true;
}

void BaseModelImpl::freeImageBuffer(image_buffer_t& image) const
{
    utils::freeImage(image);
}

}  // namespace rknn_cpp