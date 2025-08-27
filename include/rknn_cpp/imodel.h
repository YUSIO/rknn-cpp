#pragma once
#include <string>
#include <any>
#include "rknn_cpp/types.h"
#include <unordered_map>
namespace rknn_cpp
{
using ModelConfig = std::unordered_map<std::string, std::string>;

// 纯接口类
class IModel
{
   public:
    virtual ~IModel() = default;

    // 核心接口
    virtual bool initialize(const ModelConfig& config) = 0;
    virtual InferenceResult predict(const image_buffer_t& image) = 0;  // 返回统一结果
    virtual void release() = 0;

    // 信息获取接口
    virtual ModelTask getTaskType() const = 0;
    virtual std::string getModelName() const = 0;
    virtual bool isInitialized() const = 0;

    // 模型属性获取
    virtual int getModelWidth() const = 0;
    virtual int getModelHeight() const = 0;
    virtual int getModelChannels() const = 0;
};
}  // namespace rknn_cpp