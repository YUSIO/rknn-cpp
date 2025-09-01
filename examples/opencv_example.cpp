#include "rknn_cpp.h"
#include <iostream>
#include <memory>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>

int main()
{
    using namespace rknn_cpp;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "              RKNN C++ OpenCV Mat Interface Example" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 示例1: 使用ResNet进行图像分类
    {
        std::cout << "\n[EXAMPLE] ResNet Classification with cv::Mat - Batch Processing" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        auto resnet_model = createResNetModel();
        ModelConfig config = {{"model_path", "../models/resnet50-v2-7.rknn"}, {"class_file", "../models/synset.txt"}};

        // 初始化模型
        if (!resnet_model->initialize(config))
        {
            std::cerr << "Failed to initialize ResNet model" << std::endl;
            return -1;
        }

        // 获取所有JPEG图片文件
        std::string image_dir = "../inputs/imagenet/ILSVRC2012_img_val_samples";
        std::vector<std::string> image_files;

        try
        {
            for (const auto& entry : std::filesystem::directory_iterator(image_dir))
            {
                if (entry.is_regular_file())
                {
                    std::string filename = entry.path().filename().string();
                    std::string extension = entry.path().extension().string();
                    if (extension == ".JPEG" || extension == ".jpeg" || extension == ".JPG" || extension == ".jpg")
                    {
                        image_files.push_back(entry.path().string());
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error reading directory: " << e.what() << std::endl;
            resnet_model->release();
            return -1;
        }

        if (image_files.empty())
        {
            std::cerr << "No JPEG files found in " << image_dir << std::endl;
            resnet_model->release();
            return -1;
        }

        std::cout << "[INFO] Found " << image_files.size() << " images to process" << std::endl;

        // 处理每个图片
        int processed_count = 0;
        int success_count = 0;

        for (const auto& image_path : image_files)
        {
            processed_count++;

            // 提取文件名（不含路径和扩展名）
            std::string filename = std::filesystem::path(image_path).stem().string();

            std::cout << "\n[" << processed_count << "/" << image_files.size() << "] Processing: " << filename
                      << std::endl;

            // 加载图像
            cv::Mat image = cv::imread(image_path);
            if (image.empty())
            {
                std::cerr << "Failed to load image: " << image_path << std::endl;
                continue;
            }

            std::cout << "[INFO] Image size: " << image.cols << "x" << image.rows << " channels=" << image.channels()
                      << std::endl;

            // 执行推理
            auto result = resnet_model->predict(image);

            if (result.task_type == ModelTask::CLASSIFICATION)
            {
                auto classifications = result.getClassifications();
                if (!classifications.empty())
                {
                    std::cout << "[RESULT] Top predictions:" << std::endl;
                    for (size_t i = 0; i < std::min(size_t(3), classifications.size()); ++i)
                    {
                        const auto& cls = classifications[i];
                        std::cout << "        " << (i + 1) << ". " << cls.class_name << " (" << std::fixed
                                  << std::setprecision(3) << cls.confidence << ")" << std::endl;
                    }

                    // 在图像上绘制分类结果
                    cv::Mat result_image = image.clone();
                    const auto& top_cls = classifications[0];
                    std::string text = top_cls.class_name + ": " + std::to_string(top_cls.confidence).substr(0, 5);

                    // 在图像顶部绘制文本
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
                    cv::Point text_org(10, text_size.height + 10);

                    // 绘制文本背景
                    cv::rectangle(result_image, cv::Point(text_org.x - 5, text_org.y - text_size.height - 5),
                                  cv::Point(text_org.x + text_size.width + 5, text_org.y + baseline + 5),
                                  cv::Scalar(0, 0, 0), -1);

                    // 绘制文本
                    cv::putText(result_image, text, text_org, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

                    // 保存结果图像
                    std::string output_path = "../outputs/resnet_" + filename + "_result.jpg";
                    if (cv::imwrite(output_path, result_image))
                    {
                        std::cout << "[INFO] Result saved to: " << output_path << std::endl;
                        success_count++;
                    }
                    else
                    {
                        std::cerr << "[ERROR] Failed to save result to: " << output_path << std::endl;
                    }
                }
                else
                {
                    std::cerr << "[ERROR] No classification results for: " << filename << std::endl;
                }
            }
            else
            {
                std::cerr << "[ERROR] Wrong task type for: " << filename << std::endl;
            }
        }

        std::cout << "\n[SUMMARY] Processed " << processed_count << " images, " << success_count << " successful"
                  << std::endl;
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

            // 在图像上绘制检测结果
            cv::Mat result_image = image.clone();

            // 定义颜色表
            std::vector<cv::Scalar> colors = {
                cv::Scalar(255, 0, 0),    // 红色
                cv::Scalar(0, 255, 0),    // 绿色
                cv::Scalar(0, 0, 255),    // 蓝色
                cv::Scalar(255, 255, 0),  // 青色
                cv::Scalar(255, 0, 255),  // 品红色
                cv::Scalar(0, 255, 255),  // 黄色
                cv::Scalar(128, 0, 128),  // 紫色
                cv::Scalar(255, 165, 0)   // 橙色
            };

            for (size_t i = 0; i < detections.size(); ++i)
            {
                const auto& det = detections[i];
                cv::Scalar color = colors[i % colors.size()];

                // 绘制检测框
                cv::Point top_left(static_cast<int>(det.x), static_cast<int>(det.y));
                cv::Point bottom_right(static_cast<int>(det.x + det.width), static_cast<int>(det.y + det.height));
                cv::rectangle(result_image, top_left, bottom_right, color, 2);

                // 准备标签文本
                std::string label = det.class_name + ": " + std::to_string(det.confidence).substr(0, 5);

                // 计算文本大小
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

                // 绘制标签背景
                cv::Point label_top_left(top_left.x, top_left.y - text_size.height - 5);
                cv::Point label_bottom_right(top_left.x + text_size.width, top_left.y);
                cv::rectangle(result_image, label_top_left, label_bottom_right, color, -1);

                // 绘制标签文本
                cv::putText(result_image, label, cv::Point(top_left.x, top_left.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 255, 255), 1);
            }

            // 保存结果图像
            std::string output_path = "../outputs/yolov3_detection_result.jpg";
            cv::imwrite(output_path, result_image);
            std::cout << "[INFO] Detection result saved to: " << output_path << std::endl;
        }

        yolo_model->release();
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[SUCCESS] OpenCV Mat interface example completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    return 0;
}
