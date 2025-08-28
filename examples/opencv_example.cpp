#include "rknn_cpp.h"
#include <iostream>
#include <memory>
#include <iomanip>

int main()
{
    using namespace rknn_cpp;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "              RKNN C++ OpenCV Mat Interface Example" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 示例1: 使用ResNet进行图像分类
    {
        std::cout << "\n[EXAMPLE] ResNet Classification with cv::Mat" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        auto resnet_model = createResNetModel();
        ModelConfig config = {{"model_path", "../models/resnet50-v2-7.rknn"}, {"class_file", "../models/synset.txt"}};

        // 初始化模型
        if (!resnet_model->initialize(config))
        {
            std::cerr << "Failed to initialize ResNet model" << std::endl;
            return -1;
        }

        // 直接使用cv::Mat加载图像
        cv::Mat image = cv::imread("../inputs/resnet.JPEG");
        if (image.empty())
        {
            std::cerr << "Failed to load image" << std::endl;
            resnet_model->release();
            return -1;
        }

        std::cout << "[INFO] Loaded image: " << image.cols << "x" << image.rows << " channels=" << image.channels()
                  << std::endl;

        // 执行推理 - 直接使用cv::Mat接口
        auto result = resnet_model->predict(image);

        if (result.task_type == ModelTask::CLASSIFICATION)
        {
            std::cout << "\n[RESULTS] Classification Output:" << std::endl;
            std::cout << std::string(35, '-') << std::endl;

            auto classifications = result.getClassifications();
            for (size_t i = 0; i < std::min(size_t(5), classifications.size()); ++i)
            {
                const auto& cls = classifications[i];
                std::cout << "        Class " << cls.class_id << " (" << cls.class_name << "): " << std::fixed
                          << std::setprecision(3) << cls.confidence << std::endl;
            }
        }

        resnet_model->release();
    }

    // 示例2: 使用YoloV3进行目标检测
    {
        std::cout << "\n[EXAMPLE] YoloV3 Detection with cv::Mat" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        auto yolo_model = createYoloV3Model();
        ModelConfig config = {{"model_path", "../models/yolov3-tiny-i8.rknn"}};

        // 初始化模型
        if (!yolo_model->initialize(config))
        {
            std::cerr << "Failed to initialize YoloV3 model" << std::endl;
            return -1;
        }

        // 直接使用cv::Mat加载图像
        cv::Mat image = cv::imread("../inputs/image.png");
        if (image.empty())
        {
            std::cerr << "Failed to load image" << std::endl;
            yolo_model->release();
            return -1;
        }

        std::cout << "[INFO] Loaded image: " << image.cols << "x" << image.rows << " channels=" << image.channels()
                  << std::endl;

        // 执行推理 - 直接使用cv::Mat接口
        auto result = yolo_model->predict(image);

        if (result.task_type == ModelTask::OBJECT_DETECTION)
        {
            std::cout << "\n[RESULTS] Detection Output:" << std::endl;
            std::cout << std::string(35, '-') << std::endl;

            auto detections = result.getDetections();
            for (size_t i = 0; i < detections.size(); ++i)
            {
                const auto& det = detections[i];
                std::cout << "        [" << i << "] " << det.class_name << " (conf=" << std::fixed
                          << std::setprecision(3) << det.confidence << ") at (" << static_cast<int>(det.x) << ","
                          << static_cast<int>(det.y) << "," << static_cast<int>(det.width) << ","
                          << static_cast<int>(det.height) << ")" << std::endl;
            }
        }

        yolo_model->release();
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[SUCCESS] OpenCV Mat interface example completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    return 0;
}
