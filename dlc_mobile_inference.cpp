/**
 * @file dlc_mobile_inference.cpp
 * @brief 移动设备DLC模型推理脚本
 * @description 完整的两阶段网络异常检测DLC推理实现
 * 
 * 支持功能：
 * - 文件加载和保存
 * - 内存管理
 * - DLC模型加载和执行
 * - 两阶段推理流程
 * - 输出结果处理
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "src/include/json.hpp"

// SNPE Headers
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/String.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/TensorShape.hpp"

using namespace zdl;

// ================================================================================================
// 文件操作函数
// ================================================================================================

/**
 * @brief 读取文件大小
 * @param filename 文件路径
 * @return 文件大小（字节）
 */
size_t getFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return 0;
    }
    size_t size = file.tellg();
    file.close();
    return size;
}

/**
 * @brief 加载文件内容到内存
 * @param filename 文件路径
 * @param buffer 输出缓冲区
 * @param bufferSize 缓冲区大小
 * @return 是否成功
 */
bool loadFileContent(const std::string& filename, char* buffer, size_t bufferSize) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    file.read(buffer, bufferSize);
    if (file.gcount() != static_cast<std::streamsize>(bufferSize)) {
        std::cerr << "Error: Failed to read complete file " << filename << std::endl;
        file.close();
        return false;
    }
    
    file.close();
    return true;
}

/**
 * @brief 加载二进制文件到vector
 * @param filename 文件路径
 * @return 文件内容vector
 */
std::vector<uint8_t> loadBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return {};
    }
    
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
}

/**
 * @brief 保存数据到文件
 * @param filename 输出文件路径
 * @param data 数据指针
 * @param size 数据大小
 * @return 是否成功
 */
bool saveDataToFile(const std::string& filename, const void* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return false;
    }
    
    file.write(static_cast<const char*>(data), size);
    if (file.fail()) {
        std::cerr << "Error: Failed to write to file " << filename << std::endl;
        file.close();
        return false;
    }
    
    file.close();
    return true;
}

/**
 * @brief 保存输出结果到JSON文件
 * @param filename 输出文件路径
 * @param results 结果字符串
 * @return 是否成功
 */
bool saveResultsToFile(const std::string& filename, const std::string& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create results file " << filename << std::endl;
        return false;
    }
    
    file << results;
    file.close();
    return true;
}

/**
 * @brief 从JSON文件加载输入数据
 * 
 * @param filename JSON文件路径
 * @param input_vector 输出的11维向量
 * @return nlohmann::json 解析后的JSON对象
 */
nlohmann::json loadInputFromJson(const std::string& filename, std::vector<float>& input_vector) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input JSON file: " + filename);
    }
    
    nlohmann::json data = nlohmann::json::parse(file);
    
    // 验证network_data是否存在
    if (!data.contains("network_data")) {
        throw std::runtime_error("JSON is missing 'network_data' field.");
    }
    auto network_data = data["network_data"];
    
    // 按照固定顺序提取
    const std::vector<std::string> required_fields = {
        "wlan0_wireless_quality", "wlan0_signal_level", "wlan0_noise_level",
        "wlan0_rx_packets", "wlan0_tx_packets", "wlan0_rx_bytes", "wlan0_tx_bytes",
        "gateway_ping_time", "dns_resolution_time", "memory_usage_percent", "cpu_usage_percent"
    };
    
    input_vector.clear();
    for (const auto& field : required_fields) {
        if (!network_data.contains(field)) {
            throw std::runtime_error("JSON is missing field: " + field);
        }
        input_vector.push_back(network_data[field].get<float>());
    }
    
    return data;
}

// --- 数据标准化参数 (从multitask_scaler.pkl提取) ---
const std::vector<float> SCALER_MEANS = {
    73.01415281f, -51.01792166f, -88.73394555f, 14552.42279358f,
    11665.67295868f, 3023260.83382414f, 2507611.48480240f, 25.84133044f,
    42.94557904f, 42.67008283f, 27.91801190f
};
const std::vector<float> SCALER_SCALES = {
    16.12504647f, 10.82568044f, 6.66547091f, 5745.08336840f,
    4602.02848101f, 1481409.03980678f, 1211595.08803516f, 17.31558252f,
    41.49643442f, 17.50051395f, 15.07960282f
};

/**
 * @brief 对输入向量进行标准化
 * @param input_vector 需要被标准化的11维向量 (in-place)
 */
void applyStandardization(std::vector<float>& input_vector) {
    if (input_vector.size() != 11) return;
    for (size_t i = 0; i < 11; ++i) {
        input_vector[i] = (input_vector[i] - SCALER_MEANS[i]) / SCALER_SCALES[i];
    }
}


// ================================================================================================
// DLC模型管理类
// ================================================================================================

class DLCModelManager {
private:
    std::unique_ptr<zdl::DlContainer::IDlContainer> m_container;
    std::unique_ptr<zdl::SNPE::SNPE> m_snpe;
    zdl::DlSystem::TensorMap m_inputTensorMap;
    zdl::DlSystem::TensorMap m_outputTensorMap;
    std::string m_modelPath;
    
public:
    DLCModelManager() = default;
    ~DLCModelManager() = default;
    
    /**
     * @brief 加载DLC模型
     * @param modelPath DLC文件路径
     * @return 是否成功
     */
    bool loadModel(const std::string& modelPath) {
        m_modelPath = modelPath;
        
        // 1. 加载DLC容器
        std::vector<uint8_t> dlcBuffer = loadBinaryFile(modelPath);
        if (dlcBuffer.empty()) {
            std::cerr << "Error: Failed to load DLC file: " << modelPath << std::endl;
            return false;
        }
        
        m_container = zdl::DlContainer::IDlContainer::open(dlcBuffer);
        if (m_container == nullptr) {
            std::cerr << "Error: Failed to open DLC container: " << modelPath << std::endl;
            return false;
        }
        
        // 2. 创建SNPE实例
        zdl::DlSystem::RuntimeList runtimeList;
        runtimeList.add(zdl::DlSystem::Runtime_t::CPU);
        
        // 确保只使用CPU，避免在不确定的板子环境中使用DSP/GPU导致问题
        // 如果你的板子环境确认支持并配置好了GPU或DSP，可以取消下面的注释
        // runtimeList.add(zdl::DlSystem::Runtime_t::GPU_FLOAT16);
        // runtimeList.add(zdl::DlSystem::Runtime_t::DSP_FIXED8_TF);
        
        zdl::SNPE::SNPEBuilder snpeBuilder(m_container.get());
        m_snpe = snpeBuilder.setRuntimeProcessorOrder(runtimeList)
                           .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                           .build();
        
        if (m_snpe == nullptr) {
            std::cerr << "Error: Failed to create SNPE instance for: " << m_modelPath << std::endl;
            return false;
        }
        
        // --- 调试代码：打印模型找到的实际输出张量名称 ---
        auto outputTensorNames = m_snpe->getOutputTensorNames();
        if (outputTensorNames && (*outputTensorNames).size() > 0) {
            std::cout << "DEBUG: Model's actual output tensor names are:" << std::endl;
            for (const char* name : *outputTensorNames) {
                std::cout << "  -> " << name << std::endl;
            }
        } else {
            std::cout << "DEBUG: Could not retrieve any output tensor names from the model." << std::endl;
        }
        // --- 调试结束 ---

        std::cout << "Successfully loaded model: " << m_modelPath << std::endl;
        return true;
    }
    
    /**
     * @brief 获取输入张量大小
     * @return 输入张量元素数量
     */
    size_t getInputSize() {
        if (!m_snpe) return 0;
        
        auto inputTensorNames = m_snpe->getInputTensorNames();
        if (!inputTensorNames || (*inputTensorNames).size() == 0) return 0;
        
        auto inputDims = m_snpe->getInputDimensions((*inputTensorNames).at(0));
        if (!inputDims) return 0;
        
        const auto& dims = (*inputDims).getDimensions();
        size_t rank = (*inputDims).rank();
        
        size_t totalSize = 1;
        for (size_t i = 0; i < rank; ++i) {
            totalSize *= dims[i];
        }
        return totalSize;
    }
    
    /**
     * @brief 获取输出张量大小
     * @return 输出张量元素数量
     */
    size_t getOutputSize() {
        if (!m_snpe) return 0;
        
        auto outputTensorNames = m_snpe->getOutputTensorNames();
        if (!outputTensorNames || (*outputTensorNames).size() == 0) return 0;
        
        // 注意：SNPE没有getOutputDimensions方法，我们需要从输入推断或使用其他方式
        // 对于我们的模型，输出大小是已知的：检测器2个，分类器6个
        // 这里返回最大的输出大小
        return 6;  // 临时解决方案，实际应该根据模型动态获取
    }
    
    /**
     * @brief 获取指定名称的输出张量
     */
    zdl::DlSystem::ITensor* getOutputTensor(const std::string& name) {
        if (!m_snpe) return nullptr;
        return m_outputTensorMap.getTensor(name.c_str());
    }

    /**
     * @brief 执行推理
     * @param inputData 输入数据
     * @param inputSize 输入数据大小
     * @return 是否成功
     */
    bool executeInference(const float* inputData, size_t inputSize) {
        if (!m_snpe) {
            std::cerr << "Error: Model not loaded" << std::endl;
            return false;
        }
        
        // 1. 准备输入张量
        auto inputTensorNames = m_snpe->getInputTensorNames();
        if (!inputTensorNames || (*inputTensorNames).size() == 0) {
            std::cerr << "Error: No input tensor found" << std::endl;
            return false;
        }
        
        const char* inputName = (*inputTensorNames).at(0);
        
        // 获取输入维度来创建张量
        auto inputDims = m_snpe->getInputDimensions(inputName);
        if (!inputDims) {
            std::cerr << "Error: Failed to get input dimensions" << std::endl;
            return false;
        }
        
        // 使用工厂创建输入张量
        auto& tensorFactory = zdl::SNPE::SNPEFactory::getTensorFactory();
        auto inputTensor = tensorFactory.createTensor(*inputDims);
        if (inputTensor == nullptr) {
            std::cerr << "Error: Failed to create input tensor" << std::endl;
            return false;
        }
        
        // 2. 复制输入数据
        auto inputItr = inputTensor->begin();
        std::copy(inputData, inputData + inputSize, inputItr);
        
        m_inputTensorMap.add(inputName, inputTensor.release());
        
        // 2. 执行推理
        if (!m_snpe->execute(m_inputTensorMap, m_outputTensorMap)) {
            std::cerr << "Error: SNPE model execution failed" << std::endl;
            return false;
        }

        return true;
    }
    
    /**
     * @brief 清理模型资源
     */
    void cleanup() {
        m_inputTensorMap.clear();
        m_outputTensorMap.clear();
        m_snpe.reset();
        m_container.reset();
        std::cout << "Model cleanup completed" << std::endl;
    }
};

// ================================================================================================
// 输出处理函数
// ================================================================================================

/**
 * @brief 应用softmax函数
 * @param logits 输入logits
 * @param size 数组大小
 * @param probabilities 输出概率
 */
void applySoftmax(const float* logits, size_t size, float* probabilities) {
    if (!logits || !probabilities) return;
    
    std::vector<float> exp_values(size);
    float sum_of_exp = 0.0f;
    
    // 找到最大值以提高数值稳定性
    float max_logit = *std::max_element(logits, logits + size);
    
    for (size_t i = 0; i < size; ++i) {
        exp_values[i] = std::exp(logits[i] - max_logit);
        sum_of_exp += exp_values[i];
    }
    
    for (size_t i = 0; i < size; ++i) {
        probabilities[i] = exp_values[i] / sum_of_exp;
    }
}

// 异常类型映射
const std::vector<std::string> ANOMALY_CLASSES = {
    "wifi_degradation",
    "network_latency",
    "connection_instability",
    "bandwidth_congestion",
    "system_stress",
    "dns_issues"
};

/**
 * @brief 处理阶段1（异常检测）的输出
 */
nlohmann::json processDetectionOutput(const float* logits, size_t size) {
    if (size != 2) return nullptr;

    std::vector<float> probabilities(size);
    applySoftmax(logits, size, probabilities.data());

    int predicted_class = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
    bool is_anomaly = (predicted_class == 0); // 索引0是异常
    float confidence = probabilities[predicted_class];

    nlohmann::json result;
    result["is_anomaly"] = is_anomaly;
    result["confidence"] = confidence;
    result["anomaly_probability"] = probabilities[0];
    result["normal_probability"] = probabilities[1];
    result["raw_logits"] = std::vector<float>(logits, logits + size);
    return result;
}

/**
 * @brief 处理阶段2（异常分类）的输出
 */
nlohmann::json processClassificationOutput(const float* logits, size_t size) {
    if (size != ANOMALY_CLASSES.size()) return nullptr;

    std::vector<float> probabilities(size);
    applySoftmax(logits, size, probabilities.data());

    int predicted_index = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
    std::string predicted_class = ANOMALY_CLASSES[predicted_index];
    float confidence = probabilities[predicted_index];

    nlohmann::json result;
    result["predicted_class"] = predicted_class;
    result["confidence"] = confidence;
    
    nlohmann::json class_probs;
    for (size_t i = 0; i < ANOMALY_CLASSES.size(); ++i) {
        class_probs[ANOMALY_CLASSES[i]] = probabilities[i];
    }
    result["class_probabilities"] = class_probs;
    result["raw_logits"] = std::vector<float>(logits, logits + size);
    return result;
}


// 旧的二进制处理函数，保留用于兼容性
std::string processDetectionOutput_Legacy(const float* logits) {
    float anomaly_score = logits[0];
    float normal_score = logits[1];
    bool is_anomaly = anomaly_score > normal_score;
    return "Anomaly Detection Result: Anomaly=" + std::string(is_anomaly ? "YES" : "NO") +
           " (Scores: " + std::to_string(anomaly_score) + " vs " + std::to_string(normal_score) + ")";
}

std::string processClassificationOutput_Legacy(const float* logits) {
    const char* classes[] = {"wifi_degradation", "network_latency", "connection_instability", 
                             "bandwidth_congestion", "system_stress", "dns_issues"};
    int best_class = 0;
    for (int i = 1; i < 6; ++i) {
        if (logits[i] > logits[best_class]) {
            best_class = i;
        }
    }
    return "Classification Result: " + std::string(classes[best_class]);
}

/**
 * @brief JSON处理模式主函数
 */
int run_json_mode(const std::string& model_dlc, const std::string& input_json) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. 加载并标准化输入
    std::vector<float> input_buffer;
    nlohmann::json input_data;
    try {
        input_data = loadInputFromJson(input_json, input_buffer);
        std::cout << "Input JSON loaded successfully (" << input_buffer.size() << " values)" << std::endl;
        
        // --- 应用标准化 ---
        applyStandardization(input_buffer);
        std::cout << "Input data standardized." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading input: " << e.what() << std::endl;
        return 1;
    }

    // 2. 准备模型
    DLCModelManager model;
    if (!model.loadModel(model_dlc)) {
        return 1;
    }
    
    // 3. 执行推理
    std::cout << "\n--- Executing Multi-Task Model ---" << std::endl;
    if (!model.executeInference(input_buffer.data(), input_buffer.size())) {
        return 1;
    }

    // 4. 获取并处理输出
    zdl::DlSystem::ITensor* output_tensor = model.getOutputTensor("combined_output");

    if (!output_tensor) {
        std::cerr << "Error: Could not retrieve the output tensor." << std::endl;
        return 1;
    }

    // 从迭代器中提取所有8个浮点数
    std::vector<float> combined_output;
    auto tensor_itr = output_tensor->begin();
    while (tensor_itr != output_tensor->end()) {
        combined_output.push_back(*tensor_itr);
        ++tensor_itr;
    }

    // 分割输出
    std::vector<float> detection_logits(combined_output.begin(), combined_output.begin() + 2);
    std::vector<float> classification_logits(combined_output.begin() + 2, combined_output.end());

    nlohmann::json detection_result = processDetectionOutput(detection_logits.data(), detection_logits.size());
    std::cout << "Anomaly detected: " << (detection_result["is_anomaly"].get<bool>() ? "YES" : "NO") << std::endl;
    
    nlohmann::json classification_result;
    if (detection_result["is_anomaly"].get<bool>()) {
        classification_result = processClassificationOutput(classification_logits.data(), classification_logits.size());
        std::cout << "Predicted anomaly type: " << classification_result["predicted_class"].get<std::string>() << std::endl;
    } else {
        classification_result["predicted_class"] = "normal";
        classification_result["confidence"] = 1.0;
    }

    // 5. 整合并保存最终结果
    auto end_time = std::chrono::high_resolution_clock::now();
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    nlohmann::json final_result;
    final_result["timestamp"] = input_data.value("timestamp", "N/A");
    final_result["device_id"] = input_data.value("device_id", "N/A");
    final_result["processing_time_ms"] = duration;
    final_result["anomaly_detection"] = detection_result;
    final_result["anomaly_classification"] = classification_result;
    
    std::cout << "\n=== Inference Completed ===" << std::endl;
    std::cout << "Total processing time: " << duration << " ms" << std::endl;
    
    if (saveResultsToFile("inference_results.json", final_result.dump(4))) {
        std::cout << "Results saved to: inference_results.json" << std::endl;
    }

    // 清理
    model.cleanup();
    
    std::cout << "==========================" << std::endl;
    return 0;
}


int main(int argc, char* argv[]) {
    std::cout << "=== DLC Mobile Inference System (Multi-Task Model) ===" << std::endl;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dlc> <input_json_file>" << std::endl;
        std::cerr << "Example: ./dlc_mobile_inference multitask_model.dlc example_input.json" << std::endl;
        return 1;
    }

    std::string model_dlc = argv[1];
    std::string input_file = argv[2];

    std::cout << "Model DLC: " << model_dlc << std::endl;
    std::cout << "Input Data: " << input_file << std::endl;
    std::cout << "=====================================" << std::endl;

    // 只支持JSON模式
    if (input_file.size() > 5 && input_file.substr(input_file.size() - 5) == ".json") {
        return run_json_mode(model_dlc, input_file);
    } else {
        std::cerr << "Error: Input file must be a .json file" << std::endl;
        return 1;
    }
} 