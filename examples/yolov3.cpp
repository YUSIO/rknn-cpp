#include "rknn_cpp.h"
#include "rknn_cpp/models/yolov3_model.h"
#include <iostream>
#include <memory>
#include <string.h>
#include <iomanip>

int main()
{
    using namespace rknn_cpp;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "                   YOLOV3 MODEL TEST PROGRAM" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    {
        std::cout << "\n[TEST] YOLOv3 Object Detection" << std::endl;
        std::cout << std::string(45, '-') << std::endl;

        // 直接创建 YOLOv3 模型
        auto yolo_model = std::make_unique<Yolov3Model>();

        // 配置参数
        ModelConfig config = {
            {"model_path", "../models/yolov3-tiny-i8.rknn"},  // 替换为您的模型路径
        };

        // 初始化模型
        std::cout << "\n[INIT] Initializing YOLOv3 model..." << std::endl;
        if (!yolo_model->initialize(config))
        {
            std::cerr << "❌ Failed to initialize YOLOv3 model" << std::endl;

            // 尝试不带类别文件的初始化
            std::cout << "\n[RETRY] Attempting initialization without class file..." << std::endl;
            ModelConfig simple_config = {{"model_path", "../models/yolov3.rknn"}};

            if (!yolo_model->initialize(simple_config))
            {
                std::cerr << "❌ Failed to initialize YOLOv3 model even without class file" << std::endl;
                return -1;
            }
        }

        std::cout << "[SUCCESS] YOLOv3 model initialized" << std::endl;

        // 打印模型信息
        std::cout << "\n📊 Model Information:" << std::endl;
        std::cout << "  Task Type: "
                  << (yolo_model->getTaskType() == ModelTask::OBJECT_DETECTION ? "Object Detection" : "Other")
                  << std::endl;
        std::cout << "       Model Name   : " << yolo_model->getModelName() << std::endl;
        std::cout << "       Input Size   : " << yolo_model->getModelWidth() << " x " << yolo_model->getModelHeight()
                  << " x " << yolo_model->getModelChannels() << std::endl;
        std::cout << "       Initialized  : " << (yolo_model->isInitialized() ? "Yes" : "No") << std::endl;

        // 加载测试图像
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(src_image));

        std::cout << "\n[LOAD] Loading test image..." << std::endl;
        // 尝试多个可能的图像路径
        std::vector<std::string> test_images = {"../inputs/image.png"};

        bool image_loaded = false;
        for (const auto& img_path : test_images)
        {
            if (utils::readImage(img_path.c_str(), src_image))
            {
                std::cout << "[SUCCESS] Image loaded: " << img_path << std::endl;
                image_loaded = true;
                break;
            }
            else
            {
                std::cout << "[ERROR] Failed to load: " << img_path << std::endl;
            }
        }

        if (!image_loaded)
        {
            std::cerr << "❌ Failed to load any test image" << std::endl;
            return -1;
        }

        std::cout << "[INFO] Original image dimensions: " << src_image.width << " x " << src_image.height << " x 3"
                  << std::endl;

        // 尝试调用 predict 看看会发生什么
        std::cout << "\n[PREDICT] Testing prediction method..." << std::endl;
        InferenceResult result;
        try
        {
            result = yolo_model->predict(src_image);
            std::cout << "[SUCCESS] Prediction executed (result task: " << static_cast<int>(result.task_type) << ")"
                      << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cout << "[ERROR] Prediction failed: " << e.what() << std::endl;
            return -1;
        }

        // 打印结果
        std::cout << "\n[RESULTS] Inference Analysis:" << std::endl;
        std::cout << std::string(35, '-') << std::endl;

        if (result.task_type == ModelTask::OBJECT_DETECTION)
        {
            try
            {
                auto detections = result.getDetections();
                std::cout << "[DETECT] Found " << detections.size() << " objects:" << std::endl;

                for (const auto& detection : detections)
                {
                    std::cout << "        Class: " << detection.class_name << ", Confidence: " << std::fixed
                              << std::setprecision(3) << detection.confidence << ", BBox: [" << std::fixed
                              << std::setprecision(1) << detection.x << ", " << detection.y << ", " << detection.width
                              << ", " << detection.height << "]" << std::endl;
                }

                if (detections.empty())
                {
                    std::cout << "[RESULT] No objects detected." << std::endl;
                }
            }
            catch (const std::exception& e)
            {
                std::cout << "Failed to get detection results: " << e.what() << std::endl;
            }
        }
        else
        {
            std::cout << "Unexpected result task type: " << static_cast<int>(result.task_type) << std::endl;
        }

        // 清理资源
        std::cout << "\n🧹 Cleaning up..." << std::endl;
        utils::freeImage(src_image);
        std::cout << "[CLEANUP] Source image buffer freed" << std::endl;

        std::cout << "\n🎉 YOLOv3 test completed!" << std::endl;
        return 0;
    }
}