#include "rknn_cpp.h"
#include <iostream>
#include <memory>
#include <string.h>
#include <math.h>
#include <iomanip>
int main()
{
    using namespace rknn_cpp;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "              RKNN C++ INFERENCE LIBRARY EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 示例1: 使用ResNet进行图像分类
    {
        bool ret = true;
        std::cout << "\n[EXAMPLE] ResNet Classification Test" << std::endl;
        std::cout << std::string(45, '-') << std::endl;

        auto resnet_model = createResNetModel();
        ModelConfig config = {{"model_path", "../models/resnet50-v2-7.rknn"}, {"class_file", "../models/synset.txt"}};
        // 初始化模型
        if (!resnet_model->initialize(config))
        {
            std::cerr << "Failed to initialize ResNet model" << std::endl;
            return -1;
        }
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(src_image));

        // 加载图像
        ret = utils::readImage("../inputs/resnet.JPEG", src_image);
        if (!ret)
        {
            std::cerr << "Failed to load image" << std::endl;
            resnet_model->release();
            return -1;
        }
        // 执行推理
        auto result = resnet_model->predict(src_image);

        if (result.task_type == ModelTask::CLASSIFICATION)
        {
            std::cout << "\n[RESULTS] Classification Output:" << std::endl;
            std::cout << std::string(35, '-') << std::endl;

            // 方式1：使用InferenceResult的便利方法
            auto classifications = result.getClassifications();  // 返回 std::vector<ClassificationResult>

            for (size_t i = 0; i < std::min(size_t(5), classifications.size()); ++i)
            {
                const auto& cls = classifications[i];
                std::cout << "        Class " << cls.class_id << " (" << cls.class_name << "): " << std::fixed
                          << std::setprecision(3) << cls.confidence << std::endl;
            }
        }

        // 清理资源
        resnet_model->release();
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[SUCCESS] Example completed successfully!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    return 0;
}
