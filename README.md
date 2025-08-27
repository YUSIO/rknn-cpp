# RKNN C++ 推理库

一个现代化的C++17 RKNN推理库，提供了简洁的API和灵活的架构来支持各种深度学习模型的推理。

## 特性

- **现代C++设计**: 使用C++17标准，提供类型安全和性能优化
- **分层架构**: 清晰的接口分离，易于扩展和维护
- **多模型支持**: 内置ResNet分类和YOLO检测模型，支持自定义模型
- **图像处理工具**: 完整的图像预处理工具链，支持OpenCV集成
- **内存安全**: 自动资源管理和错误处理
- **跨平台**: 支持ARM64和ARMv7架构

## 架构概览

```
rknn_cpp/
├── IModel          # 模型接口
├── BaseModelImpl   # 基础实现类
├── ResNetModel     # ResNet分类模型
├── YOLOModel       # YOLO检测模型
└── utils/          # 图像处理工具
    └── image_utils # 图像预处理功能
```

## 快速开始

### 1. 构建项目

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 2. 基本使用

```cpp
#include "rknn_cpp.h"
using namespace rknn_cpp;

// 创建ResNet模型
auto model = createResNetModel();
model->initialize("resnet50.rknn");

// 加载图像
auto image = utils::readImage("test.jpg");

// 执行推理
auto result = model->predict(image);

// 处理结果
if (result.success && result.task == ModelTask::CLASSIFICATION) {
    auto& classification = result.data.classification;
    for (size_t i = 0; i < classification.class_ids.size(); ++i) {
        std::cout << "Class " << classification.class_ids[i] 
                  << ": " << classification.scores[i] << std::endl;
    }
}

// 清理
utils::freeImage(image);
model->release();
```

### 3. 自定义模型

```cpp
class MyCustomModel : public BaseModelImpl {
public:
    ModelTask getTaskType() const override {
        return ModelTask::CLASSIFICATION;
    }
    
    std::string getModelName() const override {
        return "MyModel";
    }

protected:
    bool setupModel() override {
        // 模型特定的设置
        return true;
    }
    
    bool preprocessImage(const image_buffer_t& src, image_buffer_t& dst) override {
        // 使用基类提供的帮助方法
        return standardPreprocess(src, dst);
    }
    
    InferenceResult postprocessOutputs(rknn_output* outputs, int count) override {
        // 自定义后处理逻辑
        return createClassificationResult({});
    }
};
```

## API 参考

### 核心接口

#### IModel
- `initialize(const std::string& model_path)` - 初始化模型
- `predict(const image_buffer_t& image)` - 执行推理
- `release()` - 释放资源
- `getTaskType()` - 获取任务类型
- `getModelName()` - 获取模型名称

#### BaseModelImpl
提供了RKNN集成的基础实现和帮助方法：
- `standardPreprocess()` - 标准图像预处理
- `letterboxPreprocess()` - letterbox预处理
- `createModelSizedBuffer()` - 创建模型尺寸缓冲区
- `freeImageBuffer()` - 释放图像缓冲区

### 工具函数

#### 图像处理 (utils namespace)
- `readImage(const std::string& path)` - 读取图像
- `convertImage()` - 图像格式转换
- `letterboxResize()` - letterbox缩放
- `cropAndScaleRGB()` - 裁剪和缩放
- `freeImage()` - 释放图像

#### 工厂函数
- `createResNetModel()` - 创建ResNet模型
- `createYOLOModel()` - 创建YOLO模型
- `createModel(ModelTask)` - 根据任务类型创建模型

## 模型支持

### ResNet 分类模型
- 支持ImageNet预训练模型
- 自动softmax后处理
- Top-K结果输出

### YOLO 检测模型
- 支持YOLOv5/YOLOv8系列
- NMS后处理
- 置信度过滤

## 依赖项

### 必需
- C++17编译器
- CMake 3.16+
- RKNN2 运行时库

### 可选
- OpenCV (用于图像I/O)

## 目录结构

```
rknn-cpp/
├── include/                 # 头文件
│   ├── rknn_cpp.h          # 统一头文件
│   └── rknn_cpp/
│       ├── types.h         # 类型定义
│       ├── imodel.h        # 模型接口
│       ├── base/           # 基础实现
│       ├── models/         # 模型实现
│       └── utils/          # 工具类
├── src/                    # 源文件
│   ├── base/
│   ├── models/
│   └── utils/
├── examples/               # 示例代码
├── 3rdparty/              # 第三方库
│   └── rknpu2/            # RKNN2库
└── CMakeLists.txt         # 构建配置
```

## 贡献

欢迎提交问题和贡献代码！请遵循以下指南：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持ResNet和YOLO模型
- 完整的图像处理工具链
- 可扩展的架构设计
