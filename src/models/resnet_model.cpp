#include "rknn_cpp/models/resnet_model.h"
#include "rknn_cpp/utils/image_utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace rknn_cpp
{

ResNetModel::ResNetModel() : class_names_loaded_(false) {}

ModelTask ResNetModel::getTaskType() const
{
    return ModelTask::CLASSIFICATION;
}

std::string ResNetModel::getModelName() const
{
    return "ResNet";
}

bool ResNetModel::setupModel(const ModelConfig& config)
{
    // ResNet模型的特定设置 - 与resnet50项目风格一致
    std::cout << "\n[SETUP] Configuring ResNet model parameters" << std::endl;

    const auto& input_attrs = getInputAttrs();
    const auto& output_attrs = getOutputAttrs();

    if (input_attrs.empty() || output_attrs.empty())
    {
        std::cerr << "Invalid model tensors" << std::endl;
        return false;
    }
    // 加载类别文件
    auto class_file_it = config.find("class_file");
    if (class_file_it != config.end() && !class_file_it->second.empty())
    {
        if (!loadClassNames(class_file_it->second))
        {
            std::cout << "[WARN] Failed to load class names: " << class_file_it->second << std::endl;
        }
    }

    if (class_names_loaded_)
    {
        std::cout << "[INFO] Class names loaded: " << class_names_.size() << " classes" << std::endl;
    }
    else
    {
        std::cout << "[INFO] Using default class names (no file provided)" << std::endl;
    }

    return true;
}

bool ResNetModel::preprocessImage(const image_buffer_t& src_img, image_buffer_t& dst_img)
{
    std::cout << "\n[PREPROCESS] ResNet image preprocessing" << std::endl;

    // 使用基类提供的标准预处理方法
    if (!standardPreprocess(src_img, dst_img))
    {
        std::cerr << "Failed to preprocess image" << std::endl;
        return false;
    }

    std::cout << "[INFO] Preprocessed dimensions: " << dst_img.width << " x " << dst_img.height << std::endl;
    return true;
}

bool ResNetModel::preprocessImage(const cv::Mat& src_img, cv::Mat& dst_img)
{
    std::cout << "\n[PREPROCESS] ResNet image preprocessing (cv::Mat)" << std::endl;

    // 使用基类提供的标准预处理方法
    if (!standardPreprocess(src_img, dst_img))
    {
        std::cerr << "Failed to preprocess image" << std::endl;
        return false;
    }

    std::cout << "[INFO] Preprocessed dimensions: " << dst_img.cols << " x " << dst_img.rows << std::endl;
    return true;
}

InferenceResult ResNetModel::postprocessOutputs(rknn_output* outputs, int output_count)
{
    std::cout << "\n[POSTPROCESS] ResNet classification analysis" << std::endl;

    if (outputs == nullptr || output_count <= 0)
    {
        std::cerr << "Invalid outputs" << std::endl;
        return createClassificationResult({});
    }

    const auto& output_attrs = getOutputAttrs();
    if (output_attrs.empty())
    {
        std::cerr << "No output attributes available" << std::endl;
        return createClassificationResult({});
    }

    // 获取输出数据

    int8_t* output_data = static_cast<int8_t*>(outputs[0].buf);
    if (output_data == nullptr)
    {
        std::cerr << "Output buffer is null" << std::endl;
        return createClassificationResult({});
    }

    int num_classes = output_attrs[0].n_elems;
    std::cout << "[INFO] Processing " << num_classes << " classification classes" << std::endl;
    float scale = output_attrs[0].scale;
    int32_t zp = output_attrs[0].zp;
    std::vector<float> float_output(num_classes);
    for (int i = 0; i < num_classes; i++)
    {
        float_output[i] = deqnt_affine_to_f32(output_data[i], zp, scale);
    }
    // 应用softmax
    applySoftmax(float_output.data(), num_classes);

    // 获取TopK结果 - 与resnet50项目一致
    ClassificationResults results = getTopK(float_output.data(), num_classes, 5);

    std::cout << "[RESULT] Found " << results.size() << " classification results" << std::endl;
    return createClassificationResult(results);
}

void ResNetModel::applySoftmax(float* data, int size)
{
    // 找到最大值以避免溢出 - 与resnet50项目完全一致
    float max_val = data[0];
    for (int i = 1; i < size; i++)
    {
        if (data[i] > max_val)
        {
            max_val = data[i];
        }
    }

    // 减去最大值并计算指数 - 与resnet50项目完全一致
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        data[i] = expf(data[i] - max_val);  // 使用expf而不是std::exp
        sum += data[i];
    }

    // 归一化 - 与resnet50项目完全一致
    for (int i = 0; i < size; i++)
    {
        data[i] /= sum;
    }
}

ClassificationResults ResNetModel::getTopK(const float* data, int size, int k)
{
    // 创建索引-值对的向量
    std::vector<std::pair<float, int>> elements;
    elements.reserve(size);

    for (int i = 0; i < size; i++)
    {
        elements.emplace_back(data[i], i);
    }

    // 按值降序排列 (部分排序)
    k = std::min(k, size);
    std::partial_sort(elements.begin(), elements.begin() + k, elements.end(), std::greater<std::pair<float, int>>());

    // 构建结果
    ClassificationResults results;
    results.reserve(k);

    for (int i = 0; i < k; i++)
    {
        ClassificationResult result;
        result.confidence = elements[i].first;
        result.class_id = elements[i].second;
        result.class_name = getClassName(result.class_id);  // 使用实际类名
        results.push_back(result);
    }

    return results;
}

bool ResNetModel::loadClassNames(const std::string& file_path)
{
    std::cout << "\n[LOAD] Loading class names from: " << file_path << std::endl;

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open class names file: " << file_path << std::endl;
        return false;
    }

    class_names_.clear();
    std::string line;
    int line_number = 0;

    while (std::getline(file, line))
    {
        // 去除行尾的换行符和空格
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty())
        {
            // 空行使用默认类名
            class_names_.push_back("class_" + std::to_string(line_number));
        }
        else
        {
            class_names_.push_back(line);
        }

        line_number++;
    }

    file.close();

    if (class_names_.empty())
    {
        std::cerr << "No class names loaded from file" << std::endl;
        return false;
    }

    class_names_loaded_ = true;
    std::cout << "[SUCCESS] Loaded " << class_names_.size() << " class names" << std::endl;

    // 打印前几个类名作为验证
    for (size_t i = 0; i < std::min(size_t(5), class_names_.size()); ++i)
    {
        std::cout << "        [" << i << "] " << class_names_[i] << std::endl;
    }

    return true;
}

std::string ResNetModel::getClassName(int class_id) const
{
    if (class_names_loaded_ && class_id >= 0 && class_id < static_cast<int>(class_names_.size()))
    {
        return class_names_[class_id];
    }
    else
    {
        // 如果没有加载类名文件或ID超出范围，返回默认名称
        return "class_" + std::to_string(class_id);
    }
}

float ResNetModel::deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) const
{
    return scale * ((float)qnt - (float)zp);
}
}  // namespace rknn_cpp
