# 📱 移动设备DLC部署指南

## 🎯 概述

本指南详细说明如何在移动设备/开发板上部署和运行DLC模型进行网络异常检测。我们提供了完整的C++推理实现，包含您所需的所有组件。

---

## 🏗️ 系统架构

### 📊 完整流程图
```
输入数据 → 文件加载 → 内存分配 → DLC模型推理 → 输出处理 → 结果保存
   ↓          ↓         ↓          ↓            ↓         ↓
 .bin文件 → loadFile → 缓冲区管理 → SNPE执行 → softmax → JSON输出
```

### 🔧 核心组件

正如您所说，我们的脚本包含以下完整组件：

#### 📁 **文件操作函数**
- ✅ `getFileSize()` - 读取文件大小
- ✅ `loadFileContent()` - 加载文件内容到内存
- ✅ `loadBinaryFile()` - 加载二进制文件
- ✅ `saveDataToFile()` - 保存数据到文件
- ✅ `saveResultsToFile()` - 保存JSON结果

#### 🧠 **主程序流程**
- ✅ 获取输出缓冲区大小 (`getOutputSize()`)
- ✅ 分配输入缓冲区 (`std::vector<float> inputBuffer`)
- ✅ 分配输出缓冲区 (`std::vector<float> outputBuffer`)
- ✅ 加载输入数据 (`loadFileContent()`)
- ✅ 执行模型推理 (`executeInference()`)
- ✅ 保存输出数据 (`saveDataToFile()`)
- ✅ 清理模型资源 (`cleanup()`)

---

## 📂 项目文件结构

```
📦 DLC移动部署包
├── 📄 dlc_mobile_inference.cpp      # 主要C++推理程序
├── 📄 build_mobile_inference.sh     # 编译脚本
├── 📄 generate_test_input.py        # 测试数据生成工具
├── 📄 MOBILE_DEPLOYMENT_GUIDE.md    # 本部署指南
├── 📁 DLC模型文件/
│   ├── realistic_end_to_end_anomaly_detector.dlc    (57.1 KB)
│   ├── realistic_end_to_end_anomaly_classifier.dlc  (190.2 KB)
│   └── realistic_raw_data_scaler.pkl               (0.8 KB)
├── 📁 格式规范文档/
│   ├── INPUT_FORMAT_SPECIFICATION.md
│   ├── OUTPUT_FORMAT_SPECIFICATION.md
│   └── FORMAT_SPECIFICATIONS_INDEX.md
└── 📁 测试数据/
    ├── normal_input.bin              # 正常网络状态测试数据
    ├── wifi_degradation_input.bin    # WiFi异常测试数据
    └── *.metadata.json              # 对应的元数据文件
```

---

## 🚀 快速开始

### 1️⃣ **环境准备**

```bash
# 设置SNPE SDK环境变量
export SNPE_ROOT=/path/to/snpe-2.26.2.240911

# 验证环境
echo "SNPE SDK: $SNPE_ROOT"
ls -la $SNPE_ROOT/lib/x86_64-linux-clang/libSNPE.so
```

### 2️⃣ **生成测试数据**

```bash
# 生成所有测试场景的二进制输入文件
python3 generate_test_input.py

# 或生成特定场景
python3 generate_test_input.py normal
python3 generate_test_input.py wifi_degradation
```

输出文件：
- `normal_input.bin` (44字节) - 正常网络状态
- `wifi_degradation_input.bin` (44字节) - WiFi信号衰减
- `network_latency_input.bin` (44字节) - 网络延迟异常
- `system_stress_input.bin` (44字节) - 系统压力异常

### 3️⃣ **编译推理程序**

```bash
# 设置执行权限并编译
chmod +x build_mobile_inference.sh
./build_mobile_inference.sh
```

编译成功后生成：
- `dlc_mobile_inference` - 可执行推理程序

### 4️⃣ **运行推理测试**

```bash
# 完整的两阶段推理
./dlc_mobile_inference \
    realistic_end_to_end_anomaly_detector.dlc \
    realistic_end_to_end_anomaly_classifier.dlc \
    normal_input.bin

# 测试WiFi异常检测
./dlc_mobile_inference \
    realistic_end_to_end_anomaly_detector.dlc \
    realistic_end_to_end_anomaly_classifier.dlc \
    wifi_degradation_input.bin
```

### 5️⃣ **查看结果**

```bash
# 查看JSON格式的推理结果
cat inference_results.json

# 查看原始二进制输出
ls -la stage1_output.bin stage2_output.bin
```

---

## 🔧 详细技术说明

### 📊 **内存管理详解**

我们的实现严格按照您描述的模式：

```cpp
// 1. 获取缓冲区大小
size_t inputSize = INPUT_SIZE;  // 11个float32 = 44字节
size_t outputSize = detector.getOutputSize();  // 动态获取

// 2. 分配输入缓冲区
std::vector<float> inputBuffer(inputSize);

// 3. 分配输出缓冲区  
std::vector<float> outputBuffer(outputSize);

// 4. 加载输入数据
loadFileContent(inputPath, reinterpret_cast<char*>(inputBuffer.data()), 
               inputSize * sizeof(float));

// 5. 执行推理
detector.executeInference(inputBuffer.data(), inputSize,
                         outputBuffer.data(), outputSize);

// 6. 保存输出数据
saveDataToFile("stage1_output.bin", outputBuffer.data(), 
              outputSize * sizeof(float));

// 7. 清理资源
detector.cleanup();
```

### 🎯 **两阶段推理流程**

完全符合您的要求：

#### **阶段1：异常检测**
```cpp
// 输入: 11维网络监控数据 [44字节]
// 输出: 2维logits [8字节] → softmax → 异常/正常概率
DLCModelManager detector;
detector.loadModel("realistic_end_to_end_anomaly_detector.dlc");
detector.executeInference(inputData, 11, detectionOutput, 2);
bool isAnomaly = detectionOutput[0] > detectionOutput[1];
```

#### **阶段2：异常分类（条件执行）**
```cpp
// 仅当检测到异常时执行
if (isAnomaly) {
    // 输入: 相同的11维数据
    // 输出: 6维logits [24字节] → softmax → 6种异常类型概率  
    DLCModelManager classifier;
    classifier.loadModel("realistic_end_to_end_anomaly_classifier.dlc");
    classifier.executeInference(inputData, 11, classificationOutput, 6);
}
```

### 📋 **文件操作实现**

完整实现您要求的所有文件操作：

```cpp
// 读取文件大小
size_t fileSize = getFileSize("input.bin");

// 读取文件内容到缓冲区
char* buffer = new char[fileSize];
bool success = loadFileContent("input.bin", buffer, fileSize);

// 保存输出数据
saveDataToFile("output.bin", outputData, outputSize);

// 保存JSON结果
saveResultsToFile("results.json", jsonString);
```

---

## 📱 不同平台部署

### 🖥️ **x86_64 Linux开发板**

```bash
# 编译配置
export TARGET_ARCH="x86_64-linux-clang"
./build_mobile_inference.sh
```

### 📱 **ARM64 Android设备**

```bash
# 修改编译脚本中的架构
sed -i 's/x86_64-linux-clang/arm64-android/g' build_mobile_inference.sh
export TARGET_ARCH="arm64-android"
./build_mobile_inference.sh
```

### 🎯 **ARM64 嵌入式Linux**

```bash
# 针对嵌入式Linux
export TARGET_ARCH="arm64-linux-clang"  
export CXX=aarch64-linux-gnu-g++
./build_mobile_inference.sh
```

---

## 🔬 输入输出格式详解

### 📥 **输入数据格式**

**二进制格式**: 44字节 (11个float32)
```
Offset | Size | Field                    | Range
-------|------|--------------------------|----------
0x00   | 4B   | wlan0_wireless_quality   | [0, 100]
0x04   | 4B   | wlan0_signal_level       | [-100, -10]
0x08   | 4B   | wlan0_noise_level        | [-100, -30]
0x0C   | 4B   | wlan0_rx_packets         | [0, +∞]
0x10   | 4B   | wlan0_tx_packets         | [0, +∞]
0x14   | 4B   | wlan0_rx_bytes           | [0, +∞]
0x18   | 4B   | wlan0_tx_bytes           | [0, +∞]
0x1C   | 4B   | gateway_ping_time        | [0, 5000]
0x20   | 4B   | dns_resolution_time      | [0, 5000]
0x24   | 4B   | memory_usage_percent     | [0, 100]
0x28   | 4B   | cpu_usage_percent        | [0, 100]
```

### 📤 **输出数据格式**

**阶段1输出**: 8字节 (2个float32)
```
[异常logit, 正常logit] → softmax → [异常概率, 正常概率]
```

**阶段2输出**: 24字节 (6个float32)
```
[wifi_degradation, network_latency, connection_instability,
 bandwidth_congestion, system_stress, dns_issues] → softmax → 6种异常概率
```

**JSON结果**: 结构化输出
```json
{
  "timestamp": "1704629400",
  "processing_time_ms": 12.5,
  "detection_stage": {
    "is_anomaly": true,
    "confidence": 0.8765,
    "anomaly_probability": 0.8765,
    "normal_probability": 0.1235
  },
  "classification_stage": {
    "anomaly_type": "wifi_degradation",
    "confidence": 0.7234,
    "class_probabilities": { /* 6种异常类型概率 */ }
  },
  "status": "success"
}
```

---

## 🧪 测试验证

### 🔍 **功能测试**

```bash
# 测试所有场景
for scenario in normal wifi_degradation network_latency system_stress; do
    echo "Testing $scenario..."
    ./dlc_mobile_inference \
        realistic_end_to_end_anomaly_detector.dlc \
        realistic_end_to_end_anomaly_classifier.dlc \
        ${scenario}_input.bin
    echo "Results saved to inference_results.json"
    echo "---"
done
```

### 📊 **性能测试**

```bash
# 测试推理时间
time ./dlc_mobile_inference \
    realistic_end_to_end_anomaly_detector.dlc \
    realistic_end_to_end_anomaly_classifier.dlc \
    normal_input.bin

# 查看处理时间
grep "processing_time_ms" inference_results.json
```

### 🔧 **内存使用测试**

```bash
# 使用valgrind检查内存泄漏
valgrind --leak-check=full ./dlc_mobile_inference \
    realistic_end_to_end_anomaly_detector.dlc \
    realistic_end_to_end_anomaly_classifier.dlc \
    normal_input.bin
```

---

## ⚡ 性能优化

### 🚀 **运行时优化**

```cpp
// 在DLCModelManager构造函数中选择最优运行时
DlSystem::Runtime_t runtime;
#ifdef USE_GPU
    runtime = DlSystem::Runtime_t::GPU_FLOAT16_32;  // GPU加速
#elif defined(USE_DSP)
    runtime = DlSystem::Runtime_t::DSP;             // DSP加速
#else
    runtime = DlSystem::Runtime_t::CPU;             // CPU推理
#endif
```

### 💾 **内存优化**

```cpp
// 预分配缓冲区避免重复分配
class OptimizedInference {
    std::vector<float> m_inputBuffer;
    std::vector<float> m_stage1Output;
    std::vector<float> m_stage2Output;
    
public:
    OptimizedInference() {
        // 预分配所有需要的内存
        m_inputBuffer.resize(11);
        m_stage1Output.resize(2);
        m_stage2Output.resize(6);
    }
};
```

### 📦 **模型优化**

```bash
# 量化模型以减少大小和提升速度
snpe-dlc-quantize \
    --input_dlc realistic_end_to_end_anomaly_detector.dlc \
    --input_list input_data_list.txt \
    --output_dlc detector_quantized.dlc
```

---

## 🔧 故障排除

### ❌ **常见错误及解决方案**

#### 编译错误
```bash
# 错误: SNPE headers not found
# 解决: 检查SNPE_ROOT环境变量
export SNPE_ROOT=/correct/path/to/snpe-sdk
echo $SNPE_ROOT
```

#### 运行时错误
```bash
# 错误: Failed to load DLC file
# 解决: 检查DLC文件路径和权限
ls -la *.dlc
chmod 644 *.dlc
```

#### 输入数据错误
```bash
# 错误: Input file size mismatch
# 解决: 重新生成正确格式的输入数据
python3 generate_test_input.py normal
ls -la *_input.bin  # 应该是44字节
```

### 🔍 **调试技巧**

```cpp
// 在代码中添加调试输出
#define DEBUG_MODE 1
#if DEBUG_MODE
    std::cout << "Input data: ";
    for (size_t i = 0; i < inputSize; ++i) {
        std::cout << inputBuffer[i] << " ";
    }
    std::cout << std::endl;
#endif
```

---

## 📚 相关文档

### 📖 **技术规范**
- **输入格式**: `INPUT_FORMAT_SPECIFICATION.md`
- **输出格式**: `OUTPUT_FORMAT_SPECIFICATION.md`
- **格式索引**: `FORMAT_SPECIFICATIONS_INDEX.md`

### 🛠️ **开发工具**
- **格式验证**: `simple_validate_json.py`
- **输出处理**: `process_dlc_output.py`
- **数据生成**: `generate_test_input.py`

### 🧪 **测试文件**
- **示例输入**: `example_normal_input.json`
- **示例输出**: `example_dlc_outputs.json`
- **二进制测试数据**: `*_input.bin`

---

## 🎯 总结

您的描述完全正确！我们的移动设备DLC推理脚本包含了所有您提到的组件：

### ✅ **完整实现清单**

1. **文件操作函数** ✅
   - 读取文件大小和内容函数
   - 保存文件和写入内容函数

2. **主程序流程** ✅  
   - 获取输出缓冲区大小
   - 分配输入和输出缓冲区
   - 加载输入数据和执行模型
   - 保存输出数据和cleanup model

3. **两阶段架构** ✅
   - 异常检测网络 (11维→2维)
   - 异常分类网络 (11维→6维)

4. **内存管理** ✅
   - 自动缓冲区分配和释放
   - 错误处理和资源清理

5. **格式兼容** ✅
   - 输入输出格式与转换前完全一致
   - 支持标准SNPE API

**🚀 现在您可以直接在板子上运行完整的网络异常检测系统了！** 