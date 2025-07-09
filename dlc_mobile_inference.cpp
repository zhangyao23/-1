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

// ================================================================================================
// DLC模型管理类
// ================================================================================================

class DLCModelManager {
private:
    std::unique_ptr<DlContainer::IDlContainer> m_container;
    std::unique_ptr<SNPE::SNPE> m_snpe;
    DlSystem::TensorMap m_inputTensorMap;
    DlSystem::TensorMap m_outputTensorMap;
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
        
        m_container = DlContainer::IDlContainer::open(dlcBuffer);
        if (m_container == nullptr) {
            std::cerr << "Error: Failed to open DLC container: " << modelPath << std::endl;
            return false;
        }
        
        // 2. 创建SNPE实例
        DlSystem::Runtime_t runtime = DlSystem::Runtime_t::CPU;  // 可根据需要修改为GPU_FLOAT16_32或DSP
        
        m_snpe = SNPE::SNPEFactory::createSNPE(
            *m_container,
            runtime
        );
        
        if (m_snpe == nullptr) {
            std::cerr << "Error: Failed to create SNPE instance for: " << modelPath << std::endl;
            return false;
        }
        
        std::cout << "Successfully loaded model: " << modelPath << std::endl;
        return true;
    }
    
    /**
     * @brief 获取输入张量大小
     * @return 输入张量元素数量
     */
    size_t getInputSize() {
        if (!m_snpe) return 0;
        
        const auto& inputTensorNames = m_snpe->getInputTensorNames();
        if (inputTensorNames->size() == 0) return 0;
        
        const auto& inputDims = m_snpe->getInputDimensions((*inputTensorNames)[0]);
        const auto& dims = inputDims->getDimensions();
        
        size_t totalSize = 1;
        for (size_t i = 0; i < dims.size(); ++i) {
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
        
        const auto& outputTensorNames = m_snpe->getOutputTensorNames();
        if (outputTensorNames->size() == 0) return 0;
        
        const auto& outputDims = m_snpe->getOutputDimensions((*outputTensorNames)[0]);
        const auto& dims = outputDims->getDimensions();
        
        size_t totalSize = 1;
        for (size_t i = 0; i < dims.size(); ++i) {
            totalSize *= dims[i];
        }
        return totalSize;
    }
    
    /**
     * @brief 执行推理
     * @param inputData 输入数据
     * @param inputSize 输入数据大小
     * @param outputData 输出数据缓冲区
     * @param outputSize 输出数据大小
     * @return 是否成功
     */
    bool executeInference(const float* inputData, size_t inputSize, 
                         float* outputData, size_t outputSize) {
        if (!m_snpe) {
            std::cerr << "Error: Model not loaded" << std::endl;
            return false;
        }
        
        // 1. 准备输入张量
        const auto& inputTensorNames = m_snpe->getInputTensorNames();
        if (inputTensorNames->size() == 0) {
            std::cerr << "Error: No input tensor found" << std::endl;
            return false;
        }
        
        const char* inputName = (*inputTensorNames)[0];
        auto inputTensor = m_snpe->createInputTensor(inputName);
        if (inputTensor == nullptr) {
            std::cerr << "Error: Failed to create input tensor" << std::endl;
            return false;
        }
        
        // 2. 复制输入数据
        std::copy(inputData, inputData + inputSize, 
                 inputTensor->begin().dataPointer<float>());
        
        m_inputTensorMap.add(inputName, std::move(inputTensor));
        
        // 3. 执行推理
        bool success = m_snpe->execute(m_inputTensorMap, m_outputTensorMap);
        if (!success) {
            std::cerr << "Error: Inference execution failed" << std::endl;
            return false;
        }
        
        // 4. 获取输出数据
        const auto& outputTensorNames = m_snpe->getOutputTensorNames();
        if (outputTensorNames->size() == 0) {
            std::cerr << "Error: No output tensor found" << std::endl;
            return false;
        }
        
        const char* outputName = (*outputTensorNames)[0];
        auto outputTensor = m_outputTensorMap.getTensor(outputName);
        if (outputTensor == nullptr) {
            std::cerr << "Error: Failed to get output tensor" << std::endl;
            return false;
        }
        
        // 5. 复制输出数据
        std::copy(outputTensor->cbegin().dataPointer<float>(),
                 outputTensor->cbegin().dataPointer<float>() + outputSize,
                 outputData);
        
        // 6. 清理
        m_inputTensorMap.clear();
        m_outputTensorMap.clear();
        
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
    // 找到最大值以提高数值稳定性
    float maxLogit = *std::max_element(logits, logits + size);
    
    // 计算exp值的和
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        probabilities[i] = std::exp(logits[i] - maxLogit);
        sum += probabilities[i];
    }
    
    // 归一化
    for (size_t i = 0; i < size; ++i) {
        probabilities[i] /= sum;
    }
}

/**
 * @brief 处理异常检测输出
 * @param logits 异常检测logits [2]
 * @return JSON格式结果字符串
 */
std::string processDetectionOutput(const float* logits) {
    float probabilities[2];
    applySoftmax(logits, 2, probabilities);
    
    int predictedClass = (probabilities[0] > probabilities[1]) ? 0 : 1;
    bool isAnomaly = (predictedClass == 0);
    float confidence = std::max(probabilities[0], probabilities[1]);
    
    std::string result = "{\n";
    result += "  \"detection_stage\": {\n";
    result += "    \"raw_logits\": [" + std::to_string(logits[0]) + ", " + std::to_string(logits[1]) + "],\n";
    result += "    \"probabilities\": [" + std::to_string(probabilities[0]) + ", " + std::to_string(probabilities[1]) + "],\n";
    result += "    \"predicted_class\": " + std::to_string(predictedClass) + ",\n";
    result += "    \"is_anomaly\": " + (isAnomaly ? "true" : "false") + ",\n";
    result += "    \"confidence\": " + std::to_string(confidence) + ",\n";
    result += "    \"anomaly_probability\": " + std::to_string(probabilities[0]) + ",\n";
    result += "    \"normal_probability\": " + std::to_string(probabilities[1]) + "\n";
    result += "  }";
    
    return result;
}

/**
 * @brief 处理异常分类输出
 * @param logits 异常分类logits [6]
 * @return JSON格式结果字符串
 */
std::string processClassificationOutput(const float* logits) {
    const char* anomalyClasses[] = {
        "wifi_degradation",
        "network_latency", 
        "connection_instability",
        "bandwidth_congestion",
        "system_stress",
        "dns_issues"
    };
    
    float probabilities[6];
    applySoftmax(logits, 6, probabilities);
    
    int predictedIndex = std::distance(probabilities, std::max_element(probabilities, probabilities + 6));
    float confidence = probabilities[predictedIndex];
    
    std::string result = ",\n  \"classification_stage\": {\n";
    result += "    \"raw_logits\": [";
    for (int i = 0; i < 6; ++i) {
        result += std::to_string(logits[i]);
        if (i < 5) result += ", ";
    }
    result += "],\n";
    
    result += "    \"probabilities\": [";
    for (int i = 0; i < 6; ++i) {
        result += std::to_string(probabilities[i]);
        if (i < 5) result += ", ";
    }
    result += "],\n";
    
    result += "    \"predicted_class_index\": " + std::to_string(predictedIndex) + ",\n";
    result += "    \"predicted_class_name\": \"" + std::string(anomalyClasses[predictedIndex]) + "\",\n";
    result += "    \"confidence\": " + std::to_string(confidence) + ",\n";
    result += "    \"class_probabilities\": {\n";
    
    for (int i = 0; i < 6; ++i) {
        result += "      \"" + std::string(anomalyClasses[i]) + "\": " + std::to_string(probabilities[i]);
        if (i < 5) result += ",";
        result += "\n";
    }
    result += "    }\n";
    result += "  }";
    
    return result;
}

// ================================================================================================
// 主程序
// ================================================================================================

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <detector_dlc> <classifier_dlc> <input_data_file>" << std::endl;
        std::cerr << "Example: ./dlc_mobile_inference detector.dlc classifier.dlc input.bin" << std::endl;
        return -1;
    }
    
    std::string detectorPath = argv[1];
    std::string classifierPath = argv[2]; 
    std::string inputDataPath = argv[3];
    
    std::cout << "=== DLC Mobile Inference System ===" << std::endl;
    std::cout << "Detector DLC: " << detectorPath << std::endl;
    std::cout << "Classifier DLC: " << classifierPath << std::endl;
    std::cout << "Input Data: " << inputDataPath << std::endl;
    std::cout << "=====================================" << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // ================================================================================================
    // 1. 加载输入数据
    // ================================================================================================
    
    std::cout << "Loading input data..." << std::endl;
    
    // 假设输入数据是11个float32值 (44字节)
    const size_t INPUT_SIZE = 11;
    const size_t INPUT_BYTES = INPUT_SIZE * sizeof(float);
    
    if (getFileSize(inputDataPath) != INPUT_BYTES) {
        std::cerr << "Error: Input file size mismatch. Expected " << INPUT_BYTES 
                  << " bytes, got " << getFileSize(inputDataPath) << " bytes" << std::endl;
        return -1;
    }
    
    // 分配输入缓冲区
    std::vector<float> inputBuffer(INPUT_SIZE);
    if (!loadFileContent(inputDataPath, reinterpret_cast<char*>(inputBuffer.data()), INPUT_BYTES)) {
        std::cerr << "Error: Failed to load input data" << std::endl;
        return -1;
    }
    
    std::cout << "Input data loaded successfully (" << INPUT_SIZE << " values)" << std::endl;
    
    // ================================================================================================
    // 2. 阶段1：异常检测
    // ================================================================================================
    
    std::cout << "\n--- Stage 1: Anomaly Detection ---" << std::endl;
    
    DLCModelManager detector;
    if (!detector.loadModel(detectorPath)) {
        std::cerr << "Error: Failed to load detector model" << std::endl;
        return -1;
    }
    
    // 获取输出缓冲区大小
    size_t detectorOutputSize = detector.getOutputSize();
    std::cout << "Detector output size: " << detectorOutputSize << " elements" << std::endl;
    
    // 分配输出缓冲区
    std::vector<float> detectorOutput(detectorOutputSize);
    
    // 执行异常检测
    std::cout << "Executing anomaly detection..." << std::endl;
    if (!detector.executeInference(inputBuffer.data(), INPUT_SIZE, 
                                  detectorOutput.data(), detectorOutputSize)) {
        std::cerr << "Error: Anomaly detection inference failed" << std::endl;
        detector.cleanup();
        return -1;
    }
    
    // 处理检测结果
    std::string detectionResult = processDetectionOutput(detectorOutput.data());
    bool isAnomaly = detectorOutput[0] > detectorOutput[1];  // 简单判断
    
    std::cout << "Anomaly detection completed. Is anomaly: " << (isAnomaly ? "YES" : "NO") << std::endl;
    
    // 保存阶段1输出
    if (!saveDataToFile("stage1_output.bin", detectorOutput.data(), 
                       detectorOutputSize * sizeof(float))) {
        std::cout << "Warning: Failed to save stage 1 output" << std::endl;
    }
    
    // 清理检测器
    detector.cleanup();
    
    // ================================================================================================
    // 3. 阶段2：异常分类（仅在检测到异常时执行）
    // ================================================================================================
    
    std::string classificationResult = "";
    if (isAnomaly) {
        std::cout << "\n--- Stage 2: Anomaly Classification ---" << std::endl;
        
        DLCModelManager classifier;
        if (!classifier.loadModel(classifierPath)) {
            std::cerr << "Error: Failed to load classifier model" << std::endl;
            return -1;
        }
        
        // 获取输出缓冲区大小
        size_t classifierOutputSize = classifier.getOutputSize();
        std::cout << "Classifier output size: " << classifierOutputSize << " elements" << std::endl;
        
        // 分配输出缓冲区
        std::vector<float> classifierOutput(classifierOutputSize);
        
        // 执行异常分类
        std::cout << "Executing anomaly classification..." << std::endl;
        if (!classifier.executeInference(inputBuffer.data(), INPUT_SIZE,
                                        classifierOutput.data(), classifierOutputSize)) {
            std::cerr << "Error: Anomaly classification inference failed" << std::endl;
            classifier.cleanup();
            return -1;
        }
        
        // 处理分类结果
        classificationResult = processClassificationOutput(classifierOutput.data());
        
        std::cout << "Anomaly classification completed" << std::endl;
        
        // 保存阶段2输出
        if (!saveDataToFile("stage2_output.bin", classifierOutput.data(),
                           classifierOutputSize * sizeof(float))) {
            std::cout << "Warning: Failed to save stage 2 output" << std::endl;
        }
        
        // 清理分类器
        classifier.cleanup();
    } else {
        std::cout << "\n--- Stage 2: Skipped (Normal detected) ---" << std::endl;
        classificationResult = ",\n  \"classification_stage\": null";
    }
    
    // ================================================================================================
    // 4. 保存最终结果
    // ================================================================================================
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // 构建最终JSON结果
    std::string finalResult = "{\n";
    finalResult += "  \"timestamp\": \"" + std::to_string(std::time(nullptr)) + "\",\n";
    finalResult += "  \"processing_time_ms\": " + std::to_string(duration.count()) + ",\n";
    finalResult += detectionResult;
    finalResult += classificationResult;
    finalResult += ",\n  \"status\": \"success\"\n";
    finalResult += "}\n";
    
    // 保存最终结果
    if (!saveResultsToFile("inference_results.json", finalResult)) {
        std::cerr << "Warning: Failed to save final results" << std::endl;
    }
    
    // ================================================================================================
    // 5. 输出统计信息
    // ================================================================================================
    
    std::cout << "\n=== Inference Completed ===" << std::endl;
    std::cout << "Total processing time: " << duration.count() << " ms" << std::endl;
    std::cout << "Results saved to: inference_results.json" << std::endl;
    std::cout << "Raw outputs saved to: stage1_output.bin, stage2_output.bin" << std::endl;
    std::cout << "==========================" << std::endl;
    
    return 0;
} 