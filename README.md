# RKNN-CPP 推理框架

RKNN-CPP 是一个基于 Rockchip NPU 的 C++ 深度学习推理框架，为 RK3568/RK3588 等瑞芯微平台提供高性能的神经网络推理能力。该框架封装了 RKNN Runtime API，提供简洁易用的 C++ 接口，支持图像分类和目标检测等常见任务。

## 特性

- 🚀 **高性能推理** - 充分利用 Rockchip NPU 硬件加速
- 🎯 **简洁的 API** - 统一的模型接口，易于使用和扩展
- 📦 **开箱即用** - 内置 ResNet、YOLOv3 等常用模型支持
- 🔧 **灵活配置** - 支持自定义模型和预处理方式
- 🖼️ **OpenCV 集成** - 原生支持 cv::Mat 图像输入
- 📊 **完善的结果类型** - 类型安全的推理结果封装

## 支持的模型

| 模型类型 | 模型名称 | 任务类型 | 说明 |
|---------|---------|---------|------|
| ResNetModel | ResNet50 | 图像分类 | 支持 ImageNet 1000 类分类 |
| Yolov3Model | YOLOv3-Tiny | 目标检测 | 支持 COCO 80 类检测 |
| CustomModel | 自定义模型 | 可扩展 | 用户可自定义后处理 |

## 系统要求

- **平台**: RK3568 / RK3588 等 Rockchip 平台
- **操作系统**: Linux (aarch64)
- **编译器**: 支持 C++17 的编译器 (GCC 7+ 或 Clang 5+)
- **CMake**: 3.16 或更高版本
- **OpenCV**: 4.x (必需)
- **RKNN Runtime**: 已包含在 3rdparty 目录中

## 项目结构

```
rknn-cpp/
├── include/                    # 头文件目录
│   ├── rknn_cpp.h              # 统一头文件
│   └── rknn_cpp/
│       ├── types.h             # 类型定义
│       ├── imodel.h            # 模型接口
│       ├── base/               # 基类实现
│       └── models/             # 具体模型实现
├── src/                        # 源文件目录
│   ├── base/                   # 基类实现
│   └── models/                 # 模型实现
├── examples/                   # 示例代码
│   └── opencv_example.cpp      # OpenCV 示例
├── 3rdparty/                   # 第三方库
│   └── rknpu2/                 # RKNN SDK
├── models/                     # 模型文件
├── inputs/                     # 测试输入图片
├── outputs/                    # 输出结果目录
├── CMakeLists.txt              # CMake 配置文件
└── pack.sh                     # 打包脚本
```

## 编译安装

### 1. 准备交叉编译环境 (可选)

如果在 x86 主机上交叉编译，需要安装交叉编译工具链：

```bash
# Ubuntu/Debian
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

### 2. 编译项目

```bash
# 创建构建目录
mkdir build && cd build

# 配置 CMake
cmake ..

# 编译
make -j$(nproc)
```

### 3. 安装

```bash
# 安装到 install 目录
make install
```

安装完成后，文件将被安装到 `install/` 目录：
- `install/include/` - 头文件
- `install/lib/` - 库文件
- `install/bin/` - 可执行文件
- `install/models/` - 模型文件
- `install/examples/` - 示例代码

## 快速开始

### 基本使用流程

```cpp
#include "rknn_cpp.h"

using namespace rknn_cpp;

int main() {
    // 1. 创建模型
    auto model = createResNetModel();
    
    // 2. 配置并初始化
    ModelConfig config = {
        {"model_path", "models/resnet50-v2-7.rknn"},
        {"class_file", "models/synset.txt"}
    };
    
    if (!model->initialize(config)) {
        std::cerr << "模型初始化失败" << std::endl;
        return -1;
    }
    
    // 3. 加载图像并推理
    cv::Mat image = cv::imread("test.jpg");
    auto result = model->predict(image);
    
    // 4. 处理结果
    if (result.task_type == ModelTask::CLASSIFICATION) {
        auto classifications = result.getClassifications();
        for (const auto& cls : classifications) {
            std::cout << cls.class_name << ": " << cls.confidence << std::endl;
        }
    }
    
    // 5. 释放资源
    model->release();
    
    return 0;
}
```

### 图像分类示例 (ResNet)

```cpp
#include "rknn_cpp.h"

using namespace rknn_cpp;

int main() {
    // 创建 ResNet 模型
    auto resnet = createResNetModel();
    
    ModelConfig config = {
        {"model_path", "models/resnet50-v2-7.rknn"},
        {"class_file", "models/synset.txt"}
    };
    
    if (!resnet->initialize(config)) {
        std::cerr << "ResNet 初始化失败" << std::endl;
        return -1;
    }
    
    // 使用 OpenCV 加载图像
    cv::Mat image = cv::imread("cat.jpg");
    
    // 执行推理
    auto result = resnet->predict(image);
    
    // 获取 Top-K 分类结果
    auto classifications = result.getClassifications();
    std::cout << "分类结果:" << std::endl;
    for (size_t i = 0; i < classifications.size(); ++i) {
        const auto& cls = classifications[i];
        std::cout << (i + 1) << ". " << cls.class_name 
                  << " (置信度: " << cls.confidence << ")" << std::endl;
    }
    
    resnet->release();
    return 0;
}
```

### 目标检测示例 (YOLOv3)

```cpp
#include "rknn_cpp.h"

using namespace rknn_cpp;

int main() {
    // 创建 YOLOv3 模型
    auto yolo = createYoloV3Model();
    
    ModelConfig config = {
        {"model_path", "models/yolov3-tiny-i8.rknn"}
    };
    
    if (!yolo->initialize(config)) {
        std::cerr << "YOLOv3 初始化失败" << std::endl;
        return -1;
    }
    
    // 加载图像
    cv::Mat image = cv::imread("street.jpg");
    
    // 执行推理
    auto result = yolo->predict(image);
    
    // 获取检测结果
    auto detections = result.getDetections();
    std::cout << "检测到 " << detections.size() << " 个目标:" << std::endl;
    
    for (const auto& det : detections) {
        std::cout << "- " << det.class_name 
                  << " [置信度: " << det.confidence << "]"
                  << " 位置: (" << det.x << ", " << det.y 
                  << ", " << det.width << ", " << det.height << ")" 
                  << std::endl;
        
        // 在图像上绘制检测框
        cv::rectangle(image, 
                      cv::Point(det.x, det.y),
                      cv::Point(det.x + det.width, det.y + det.height),
                      cv::Scalar(0, 255, 0), 2);
    }
    
    cv::imwrite("detection_result.jpg", image);
    
    yolo->release();
    return 0;
}
```

### 使用工厂方法创建模型

```cpp
#include "rknn_cpp.h"

using namespace rknn_cpp;

int main() {
    // 根据任务类型创建模型
    auto classifier = createModel(ModelTask::CLASSIFICATION);
    auto detector = createModel(ModelTask::OBJECT_DETECTION);
    
    // 或者直接使用具体的工厂方法
    auto resnet = createResNetModel();
    auto yolo = createYoloV3Model();
    auto custom = createCustomModel();
    
    return 0;
}
```

## API 参考

### 模型接口 (IModel)

```cpp
class IModel {
public:
    // 初始化模型
    virtual bool initialize(const ModelConfig& config) = 0;
    
    // 执行推理
    virtual InferenceResult predict(const cv::Mat& image) = 0;
    
    // 释放资源
    virtual void release() = 0;
    
    // 获取模型信息
    virtual ModelTask getTaskType() const = 0;
    virtual std::string getModelName() const = 0;
    virtual bool isInitialized() const = 0;
    
    // 获取模型属性
    virtual int getModelWidth() const = 0;
    virtual int getModelHeight() const = 0;
    virtual int getModelChannels() const = 0;
};
```

### 模型配置 (ModelConfig)

ModelConfig 是一个 `std::unordered_map<std::string, std::string>` 类型，支持以下配置项：

| 配置项 | 说明 | 示例 |
|-------|------|------|
| `model_path` | RKNN 模型文件路径 | `"models/resnet50.rknn"` |
| `class_file` | 类别名称文件路径 | `"models/synset.txt"` |

### 推理结果 (InferenceResult)

```cpp
struct InferenceResult {
    ModelTask task_type;       // 任务类型
    std::any result_data;      // 结果数据
    bool is_success;           // 是否成功
    float inference_time;      // 推理时间
    float total_time;          // 总处理时间
    
    // 便利方法
    DetectionResults getDetections() const;
    ClassificationResults getClassifications() const;
};
```

### 分类结果 (ClassificationResult)

```cpp
struct ClassificationResult {
    uint8_t class_id;          // 类别 ID
    std::string class_name;    // 类别名称
    float confidence;          // 置信度
};
```

### 检测结果 (DetectionResult)

```cpp
struct DetectionResult {
    uint16_t x, y;             // 边界框左上角坐标
    uint16_t width, height;    // 边界框宽高
    float confidence;          // 置信度
    uint16_t class_id;         // 类别 ID
    std::string class_name;    // 类别名称
};
```

## 运行示例

编译完成后，可以运行内置示例：

```bash
cd build

# 运行 OpenCV 示例
./opencv_example
```

示例程序会自动处理 `inputs/` 目录中的图片，并将结果保存到 `outputs/` 目录。

## 自定义模型扩展

如需添加新的模型支持，可以继承 `BaseModelImpl` 基类：

```cpp
#include "rknn_cpp/base/base_model_impl.h"

class MyCustomModel : public BaseModelImpl {
public:
    ModelTask getTaskType() const override {
        return ModelTask::CLASSIFICATION;
    }
    
    std::string getModelName() const override {
        return "MyCustomModel";
    }

protected:
    bool setupModel(const ModelConfig& config) override {
        // 加载模型和配置
        return loadRKNNModel(config.at("model_path"));
    }
    
    bool preprocessImage(const cv::Mat& src, cv::Mat& dst) override {
        // 自定义预处理
        return standardPreprocess(src, dst);
    }
    
    InferenceResult postprocessOutputs(rknn_output* outputs, 
                                       int output_count) override {
        // 自定义后处理
        ClassificationResults results;
        // ... 解析输出 ...
        return createClassificationResult(results);
    }
};
```

## 许可证

本项目仅供学习和研究使用。

## 致谢

- [Rockchip](https://www.rock-chips.com/) - RKNN SDK
- [OpenCV](https://opencv.org/) - 图像处理库
