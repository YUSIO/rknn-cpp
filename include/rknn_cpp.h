#pragma once

/**
 * @file rknn_cpp.h
 * @brief RKNN C++ 推理库统一头文件
 *
 * 这个头文件包含了RKNN C++推理库的所有公共API，包括：
 * - 核心类型定义
 * - 模型接口
 * - 具体模型实现
 * - 图像处理工具
 *
 * 使用方法：
 * ```cpp
 * #include "rknn_cpp.h"
 * using namespace rknn_cpp;
 *
 * // 创建ResNet模型
 * auto model = std::make_unique<ResNetModel>();
 * model->initialize("resnet50.rknn");
 *
 * // 使用image_buffer_t
 * auto image = utils::readImage("image.jpg");
 * auto result = model->predict(image);
 *
 * // 或者直接使用cv::Mat
 * cv::Mat mat = cv::imread("image.jpg");
 * auto result2 = model->predict(mat);
 *
 * model->release();
 * ```
 */

// OpenCV支持
#include <opencv2/opencv.hpp>

// 核心类型定义
#include "rknn_cpp/types.h"

// 模型接口
#include "rknn_cpp/imodel.h"

// 基础实现
#include "rknn_cpp/base/base_model_impl.h"

// 具体模型实现
#include "rknn_cpp/models/resnet_model.h"
#include "rknn_cpp/models/yolov3_model.h"

// 工具类
#include "rknn_cpp/utils/image_utils.h"

/**
 * @namespace rknn_cpp
 * @brief RKNN C++ 推理库命名空间
 *
 * 包含了所有的模型、工具和类型定义
 */
namespace rknn_cpp
{

/**
 * @brief 创建ResNet分类模型
 * @return ResNet模型的唯一指针
 *
 * @example
 * ```cpp
 * auto model = createResNetModel();
 * model->initialize("resnet50.rknn");
 * ```
 */
inline std::unique_ptr<IModel> createResNetModel()
{
    return std::make_unique<ResNetModel>();
}

inline std::unique_ptr<IModel> createYoloV3Model()
{
    return std::make_unique<Yolov3Model>();
}
/**
 * @brief 根据任务类型创建模型
 * @param task 模型任务类型
 * @return 对应的模型实例，如果任务类型不支持则返回nullptr
 */
inline std::unique_ptr<IModel> createModel(ModelTask task)
{
    switch (task)
    {
        case ModelTask::CLASSIFICATION:
            return createResNetModel();
        case ModelTask::OBJECT_DETECTION:
            return createYoloV3Model();
        default:
            return nullptr;
    }
}

}  // namespace rknn_cpp
