#include "rknn_cpp/models/yolov3_model.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <set>
#include <numeric>
#include <iomanip>

namespace rknn_cpp
{
static const int PROP_BOX_SIZE = 6;  // 4(bbox) + 1(conf) + 80(classes)
static const int OBJ_CLASS_NUM = 1;  // COCO数据集类别数

Yolov3Model::Yolov3Model() : class_names_loaded_(false) {}
ModelTask Yolov3Model::getTaskType() const
{
    return ModelTask::OBJECT_DETECTION;
}

std::string rknn_cpp::Yolov3Model::getModelName() const
{
    return "Yolov3";
}

bool Yolov3Model::setupModel(const ModelConfig& config)
{
    std::cout << "Setting up Yolov3 model..." << std::endl;
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
    auto conf_threshold = config.find("conf_threshold");
    if (conf_threshold != config.end() && !conf_threshold->second.empty())
    {
        this->conf_threshold_ = stof(conf_threshold->second);
    }
    else
    {
        this->conf_threshold_ = 0.25f;
    }
    auto nms_threshold = config.find("nms_threshold");
    if (nms_threshold != config.end() && !nms_threshold->second.empty())
    {
        this->nms_threshold_ = stof(nms_threshold->second);
    }
    else
    {
        this->nms_threshold_ = 0.1f;
    }
    return true;
}

bool Yolov3Model::preprocessImage(const cv::Mat& src_img, cv::Mat& dst_img)
{
    cv::Mat input_img{};

    if (src_img.channels() == 1 && getModelChannels() == 3)
    {
        cv::cvtColor(src_img, input_img, cv::COLOR_GRAY2RGB);
    }
    else
    {
        input_img = src_img;
    }
    std::cout << "\n[PREPROCESS] YOLOv3 image preprocessing (cv::Mat)" << std::endl;

    // 计算缩放比例，保持长宽比
    float scale_x = static_cast<float>(getModelWidth()) / input_img.cols;
    float scale_y = static_cast<float>(getModelHeight()) / input_img.rows;
    letterbox_params_.scale = std::min(scale_x, scale_y);

    int scaled_width = static_cast<int>(input_img.cols * letterbox_params_.scale);
    int scaled_height = static_cast<int>(input_img.rows * letterbox_params_.scale);

    // 计算居中位置
    letterbox_params_.x_pad = (getModelWidth() - scaled_width) / 2;
    letterbox_params_.y_pad = (getModelHeight() - scaled_height) / 2;

    // 创建目标图像并填充背景色
    dst_img = cv::Mat(getModelHeight(), getModelWidth(), CV_8UC3, cv::Scalar(144, 144, 144));

    // 缩放源图像
    cv::Mat resized;
    cv::resize(input_img, resized, cv::Size(scaled_width, scaled_height));

    // 将缩放后的图像复制到目标图像的中心位置
    cv::Rect roi(letterbox_params_.x_pad, letterbox_params_.y_pad, scaled_width, scaled_height);
    resized.copyTo(dst_img(roi));

    std::cout << "[INFO] Preprocessed dimensions: " << dst_img.cols << " x " << dst_img.rows << std::endl;
    std::cout << "[INFO] Letterbox params - scale: " << letterbox_params_.scale
              << ", x_pad: " << letterbox_params_.x_pad << ", y_pad: " << letterbox_params_.y_pad << std::endl;

    return true;
}

InferenceResult Yolov3Model::postprocessOutputs(rknn_output* outputs, int output_count)
{
    std::cout << "\n[POSTPROCESS] YOLOv3 detection analysis" << std::endl;

    if (outputs == nullptr || output_count <= 0)
    {
        std::cerr << "Invalid outputs for postprocessing" << std::endl;
        return createEmptyResult();
    }

    // YOLOv3的三个检测层配置
    // std::vector<YoloLayer> yolo_layers = {
    //     {40, 40, 16, {10, 14, 23, 27, 37, 58}},      // 中目标检测层
    //     {20, 20, 32, {81, 82, 135, 169, 344, 319}},  // 大目标检测层
    // };
    std::vector<YoloLayer> yolo_layers = {{40, 40, 16, {3.59968, 3.59968, 4.5352, 3.80864, 4.55072, 4.54688}},
                                          {20, 20, 32, {5.34368, 4.57824, 4.81248, 5.6016, 6.67584, 5.71488}}};
    const auto& output_attrs = getOutputAttrs();
    std::vector<float> boxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    int total_valid_boxes = 0;

    // 处理每个输出层
    for (int i = 0; i < output_count && i < static_cast<int>(yolo_layers.size()); ++i)
    {
        const auto& attr = output_attrs[i];
        const auto& layer = yolo_layers[i];

        std::cout << "[LAYER " << i << "] Processing output: " << layer.grid_h << " x " << layer.grid_w
                  << " (stride=" << layer.stride << ")" << std::endl;

        // 验证输出维度
        if (attr.dims[2] != static_cast<uint32_t>(layer.grid_h) || attr.dims[3] != static_cast<uint32_t>(layer.grid_w))
        {
            std::cerr << "Warning: Output dimensions mismatch for layer " << i << ", expected " << layer.grid_h << "x"
                      << layer.grid_w << ", got " << attr.dims[2] << "x" << attr.dims[3] << std::endl;
        }

        int valid_count = 0;

        if (isQuantized())
        {
            std::cout << "[QUANT] zp=" << attr.zp << ", scale=" << std::fixed << std::setprecision(6) << attr.scale
                      << std::endl;
            std::cout << "[MODE] Processing quantized model" << std::endl;
            // 处理量化模型
            valid_count = processYoloLayer(outputs[i].buf, true, layer, boxes, objProbs, classId, this->conf_threshold_,
                                           attr.zp, attr.scale);
        }
        else
        {
            // 处理浮点模型
            std::cout << "Process float model" << std::endl;
            valid_count =
                processYoloLayer(outputs[i].buf, false, layer, boxes, objProbs, classId, this->conf_threshold_);
        }

        total_valid_boxes += valid_count;
    }

    std::cout << "\n[NMS] Pre-filtering summary" << std::endl;
    std::cout << "      Total detections: " << total_valid_boxes << std::endl;

    // 应用NMS
    std::vector<int> keep_indices = applyNMS(boxes, objProbs, classId, this->nms_threshold_);

    // 构建最终的检测结果
    DetectionResults detections;
    detections.reserve(keep_indices.size());

    for (int idx : keep_indices)
    {
        DetectionResult detection;
        detection.class_id = static_cast<uint16_t>(classId[idx]);
        detection.class_name = getClassName(classId[idx]);
        detection.confidence = objProbs[idx];
        detection.x = static_cast<uint16_t>(round(boxes[idx * 4]));
        detection.y = static_cast<uint16_t>(round(boxes[idx * 4 + 1]));
        detection.width = static_cast<uint16_t>(round(boxes[idx * 4 + 2]));
        detection.height = static_cast<uint16_t>(round(boxes[idx * 4 + 3]));

        detections.push_back(detection);
    }

    std::cout << "[RESULT] Final detections before coordinate conversion: " << detections.size() << std::endl;

    // 将坐标从letterbox空间转换回原始图像空间
    convertLetterboxToOriginal(detections, getOriginalWidth(), getOriginalHeight());

    std::cout << "[RESULT] Final detections after coordinate conversion: " << detections.size() << std::endl;

    // 打印每个检测项的详细信息
    std::cout << "\n[DETECTIONS] Detailed list:" << std::endl;
    for (size_t i = 0; i < detections.size(); ++i)
    {
        const auto& d = detections[i];
        std::cout << "  [" << i << "] " << d.class_name << " (id=" << d.class_id << ") "
                  << "conf=" << std::fixed << std::setprecision(3) << d.confidence << " "
                  << "bbox=(x=" << d.x << ", y=" << d.y << ", w=" << d.width << ", h=" << d.height << ")" << std::endl;
    }

    // 使用基类的便利方法创建结果
    return createDetectionResult(detections);
}  // namespace rknn_cpp

bool Yolov3Model::loadClassNames(const std::string& file_path)
{
    std::cout << "Loading class names from: " << file_path << std::endl;

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
    std::cout << "Loaded " << class_names_.size() << " class names" << std::endl;

    // 打印前几个类名作为验证
    for (size_t i = 0; i < std::min(size_t(5), class_names_.size()); ++i)
    {
        std::cout << "  " << i << ": " << class_names_[i] << std::endl;
    }

    return true;
}
float Yolov3Model::sigmoid(float x) const
{
    return 1.0f / (1.0f + expf(-x));
}
float Yolov3Model::deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) const
{
    return scale * ((float)qnt - (float)zp);
}

int Yolov3Model::processYoloLayer(void* input, bool is_quantized, const YoloLayer& layer, std::vector<float>& boxes,
                                  std::vector<float>& objProbs, std::vector<int>& classId, float threshold, int32_t zp,
                                  float scale) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int validCount = 0;
    int grid_len = layer.grid_h * layer.grid_w;

    if (is_quantized)
    {
        int8_t* data = static_cast<int8_t*>(input);
        for (int a = 0; a < 3; a++)
        {
            for (int i = 0; i < layer.grid_h; i++)
            {
                for (int j = 0; j < layer.grid_w; j++)
                {
                    int conf_idx = (PROP_BOX_SIZE * a + 4) * grid_len + i * layer.grid_w + j;
                    float conf_input = deqnt_affine_to_f32(data[conf_idx], zp, scale);
                    float box_confidence = sigmoid(conf_input);

                    if (box_confidence >= threshold)
                    {
                        int offset = (PROP_BOX_SIZE * a) * grid_len + i * layer.grid_w + j;

                        float tx = deqnt_affine_to_f32(data[offset], zp, scale);
                        float ty = deqnt_affine_to_f32(data[offset + grid_len], zp, scale);
                        float tw = deqnt_affine_to_f32(data[offset + 2 * grid_len], zp, scale);
                        float th = deqnt_affine_to_f32(data[offset + 3 * grid_len], zp, scale);

                        float sig_tx = sigmoid(tx);
                        float sig_ty = sigmoid(ty);
                        float sig_tw = sigmoid(tw);
                        float sig_th = sigmoid(th);

                        float box_x = sig_tx * 2.0f - 0.5f;
                        float box_y = sig_ty * 2.0f - 0.5f;
                        float box_w = powf(sig_tw * 2.0f, 2);
                        float box_h = powf(sig_th * 2.0f, 2);

                        box_x = (box_x + j) * static_cast<float>(layer.stride);
                        box_y = (box_y + i) * static_cast<float>(layer.stride);
                        box_w *= (layer.anchors[a * 2]);
                        box_h *= (layer.anchors[a * 2 + 1]);

                        box_x -= (box_w / 2.0f);
                        box_y -= (box_h / 2.0f);

                        float class_input = deqnt_affine_to_f32(data[offset + 5 * grid_len], zp, scale);
                        float maxClassProbs = sigmoid(class_input);
                        int maxClassId = 0;

                        for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                        {
                            float cls_input = deqnt_affine_to_f32(data[offset + (5 + k) * grid_len], zp, scale);
                            float prob = sigmoid(cls_input);
                            if (prob > maxClassProbs)
                            {
                                maxClassId = k;
                                maxClassProbs = prob;
                            }
                        }

                        float final_conf = maxClassProbs * box_confidence;
                        if (final_conf > threshold)
                        {
                            objProbs.push_back(final_conf);
                            classId.push_back(maxClassId);
                            validCount++;

                            boxes.push_back(box_x);
                            boxes.push_back(box_y);
                            boxes.push_back(box_w);
                            boxes.push_back(box_h);
                        }
                    }
                }
            }
        }
    }
    else
    {
        float* data = static_cast<float*>(input);
        for (int a = 0; a < 3; a++)
        {
            for (int i = 0; i < layer.grid_h; i++)
            {
                for (int j = 0; j < layer.grid_w; j++)
                {
                    int conf_idx = (PROP_BOX_SIZE * a + 4) * grid_len + i * layer.grid_w + j;
                    float box_confidence = sigmoid(data[conf_idx]);

                    if (box_confidence >= threshold)
                    {
                        int offset = (PROP_BOX_SIZE * a) * grid_len + i * layer.grid_w + j;

                        float tx = data[offset];
                        float ty = data[offset + grid_len];
                        float tw = data[offset + 2 * grid_len];
                        float th = data[offset + 3 * grid_len];

                        float sig_tx = sigmoid(tx);
                        float sig_ty = sigmoid(ty);
                        float sig_tw = sigmoid(tw);
                        float sig_th = sigmoid(th);

                        float box_x = sig_tx * 2.0f - 0.5f;
                        float box_y = sig_ty * 2.0f - 0.5f;
                        float box_w = powf(sig_tw * 2.0f, 2);
                        float box_h = powf(sig_th * 2.0f, 2);

                        box_x = (box_x + j) * static_cast<float>(layer.stride);
                        box_y = (box_y + i) * static_cast<float>(layer.stride);
                        box_w *= (layer.anchors[a * 2]);
                        box_h *= (layer.anchors[a * 2 + 1]);

                        box_x -= (box_w / 2.0f);
                        box_y -= (box_h / 2.0f);

                        float maxClassProbs = sigmoid(data[offset + 5 * grid_len]);
                        int maxClassId = 0;

                        for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                        {
                            float prob = sigmoid(data[offset + (5 + k) * grid_len]);
                            if (prob > maxClassProbs)
                            {
                                maxClassId = k;
                                maxClassProbs = prob;
                            }
                        }

                        float final_conf = maxClassProbs * box_confidence;
                        if (final_conf > threshold)
                        {
                            objProbs.push_back(final_conf);
                            classId.push_back(maxClassId);
                            validCount++;

                            boxes.push_back(box_x);
                            boxes.push_back(box_y);
                            boxes.push_back(box_w);
                            boxes.push_back(box_h);
                        }
                    }
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "[TIMING] Layer processed in " << duration.count() << " μs" << std::endl;
    std::cout << "[RESULT] Found " << validCount << " valid detections" << std::endl;

    return validCount;
}

float Yolov3Model::calculateIoU(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1,
                                float xmax1, float ymax1) const
{
    // 计算交集区域
    float inter_xmin = std::max(xmin0, xmin1);
    float inter_ymin = std::max(ymin0, ymin1);
    float inter_xmax = std::min(xmax0, xmax1);
    float inter_ymax = std::min(ymax0, ymax1);

    // 检查是否有交集
    if (inter_xmin >= inter_xmax || inter_ymin >= inter_ymax)
    {
        return 0.0f;
    }

    // 计算交集面积
    float inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin);

    // 计算并集面积
    float area0 = (xmax0 - xmin0) * (ymax0 - ymin0);
    float area1 = (xmax1 - xmin1) * (ymax1 - ymin1);
    float union_area = area0 + area1 - inter_area;

    // 避免除零
    if (union_area <= 0.0f)
    {
        return 0.0f;
    }

    return inter_area / union_area;
}

int Yolov3Model::nmsForClass(const std::vector<float>& boxes, const std::vector<int>& classIds, std::vector<int>& order,
                             int filterId, float threshold) const
{
    int validCount = static_cast<int>(order.size());

    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        // 跳过已被抑制的框或不是目标类别的框
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }

        // 获取当前框的坐标 (x, y, w, h)
        float xmin0 = boxes[n * 4 + 0];
        float ymin0 = boxes[n * 4 + 1];
        float xmax0 = boxes[n * 4 + 0] + boxes[n * 4 + 2];  // x + w
        float ymax0 = boxes[n * 4 + 1] + boxes[n * 4 + 3];  // y + h

        // 与后续所有框比较
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            // 跳过已被抑制的框或不是目标类别的框
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }

            // 获取比较框的坐标 (x, y, w, h)
            float xmin1 = boxes[m * 4 + 0];
            float ymin1 = boxes[m * 4 + 1];
            float xmax1 = boxes[m * 4 + 0] + boxes[m * 4 + 2];  // x + w
            float ymax1 = boxes[m * 4 + 1] + boxes[m * 4 + 3];  // y + h

            // 计算IoU
            float iou = calculateIoU(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            // 如果IoU超过阈值，抑制置信度较低的框
            if (iou > threshold)
            {
                order[j] = -1;  // 标记为被抑制
            }
        }
    }

    return 0;
}

std::vector<int> Yolov3Model::applyNMS(const std::vector<float>& boxes, const std::vector<float>& scores,
                                       const std::vector<int>& classIds, float nms_threshold) const
{
    int validCount = static_cast<int>(boxes.size() / 4);

    if (validCount == 0)
    {
        return {};
    }

    std::cout << "\n[NMS] Applying Non-Maximum Suppression" << std::endl;
    std::cout << "      Threshold: " << std::fixed << std::setprecision(3) << nms_threshold << std::endl;
    std::cout << "      Input boxes: " << validCount << std::endl;

    // 创建索引数组并按置信度排序
    std::vector<int> order(validCount);
    std::iota(order.begin(), order.end(), 0);  // 填充0,1,2...

    // 按置信度降序排序
    std::sort(order.begin(), order.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

    // 获取所有唯一的类别ID
    std::set<int> unique_classes(classIds.begin(), classIds.end());

    // 对每个类别分别进行NMS
    for (int class_id : unique_classes)
    {
        nmsForClass(boxes, classIds, order, class_id, nms_threshold);
    }

    // 收集未被抑制的检测框索引
    std::vector<int> keep_indices;
    for (int i = 0; i < validCount; ++i)
    {
        if (order[i] != -1)  // 未被抑制
        {
            keep_indices.push_back(order[i]);
        }
    }

    std::cout << "NMS completed: " << keep_indices.size() << " boxes kept out of " << validCount << std::endl;

    return keep_indices;
}
std::string Yolov3Model::getClassName(int class_id) const
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

void Yolov3Model::convertLetterboxToOriginal(DetectionResults& detections, int orig_width, int orig_height) const
{
    std::cout << "\n[LETTERBOX] Converting coordinates to original image space" << std::endl;
    std::cout << "            Original size: " << orig_width << " x " << orig_height << std::endl;
    std::cout << "            Scale: " << letterbox_params_.scale << ", Pads: (" << letterbox_params_.x_pad << ", "
              << letterbox_params_.y_pad << ")" << std::endl;

    for (auto& detection : detections)
    {
        // 保存原始坐标用于调试
        float orig_x = detection.x;
        float orig_y = detection.y;
        float orig_w = detection.width;
        float orig_h = detection.height;

        // 转换坐标：从letterbox空间转换到原始图像空间
        // 1. 减去pad偏移
        float x_no_pad = detection.x - letterbox_params_.x_pad;
        float y_no_pad = detection.y - letterbox_params_.y_pad;

        // 2. 除以scale恢复原始尺寸
        detection.x = static_cast<uint16_t>(std::max(0.0f, x_no_pad / letterbox_params_.scale));
        detection.y = static_cast<uint16_t>(std::max(0.0f, y_no_pad / letterbox_params_.scale));
        detection.width = static_cast<uint16_t>(detection.width / letterbox_params_.scale);
        detection.height = static_cast<uint16_t>(detection.height / letterbox_params_.scale);

        // 3. 确保坐标在原图范围内
        detection.x = std::min(detection.x, static_cast<uint16_t>(orig_width));
        detection.y = std::min(detection.y, static_cast<uint16_t>(orig_height));
        detection.width = std::min(detection.width, static_cast<uint16_t>(orig_width - detection.x));
        detection.height = std::min(detection.height, static_cast<uint16_t>(orig_height - detection.y));

        std::cout << "            [" << detection.class_name << "] "
                  << "(" << orig_x << "," << orig_y << "," << orig_w << "," << orig_h << ") -> "
                  << "(" << detection.x << "," << detection.y << "," << detection.width << "," << detection.height
                  << ")" << std::endl;
    }
}
}  // namespace rknn_cpp