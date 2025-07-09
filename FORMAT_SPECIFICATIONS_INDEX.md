# 📚 DLC模型格式规范索引

## 🎯 概述

本文档是DLC模型输入输出格式规范的完整索引，帮助开发者快速找到所需的格式规范和工具。

---

## 📊 输入格式规范

### 📋 主要文档
- **📄 `INPUT_FORMAT_SPECIFICATION.md`** - 完整的输入格式规范
  - 11维输入字段详细说明
  - JSON标准格式模板
  - 数据验证规则
  - Python/C++转换示例
  - 不同场景的输入示例

### 🛠️ 工具和示例
- **🔧 `simple_validate_json.py`** - JSON输入验证工具
- **📋 `example_normal_input.json`** - 正常网络状态输入示例
- **🧪 `validate_json_input.py`** - 高级验证工具（需要numpy）

### 💡 快速开始
```bash
# 验证输入格式
python3 simple_validate_json.py example_normal_input.json

# 查看输入格式规范
cat INPUT_FORMAT_SPECIFICATION.md
```

---

## 📤 输出格式规范

### 📋 主要文档
- **📄 `OUTPUT_FORMAT_SPECIFICATION.md`** - 完整的输出格式规范
  - 两阶段DLC输出处理
  - Softmax概率计算
  - 最终结果整合
  - 错误处理和验证

### 🛠️ 工具和示例
- **🔧 `process_dlc_output.py`** - DLC输出处理工具
- **📋 `example_dlc_outputs.json`** - 各种场景的DLC输出示例
- **🎯 `DLCOutputProcessor类`** - 可集成的输出处理器

### 💡 快速开始
```bash
# 处理输出格式
python3 process_dlc_output.py

# 查看输出示例
cat example_dlc_outputs.json

# 查看输出格式规范
cat OUTPUT_FORMAT_SPECIFICATION.md
```

---

## 🏗️ 系统架构概览

### 📊 数据流程图
```
11维JSON输入 → 输入验证 → DLC模型推理 → 输出处理 → 标准JSON响应
     ↓              ↓           ↓            ↓         ↓
INPUT_FORMAT → validate_json → DLC文件 → process_output → 最终结果
```

### 🔍 两阶段DLC架构
```
输入(11维) → 异常检测DLC → 2维输出 → 异常分类DLC → 6维输出 → 整合结果
          realistic_end_to_end_anomaly_detector.dlc
                                   realistic_end_to_end_anomaly_classifier.dlc
```

---

## 📋 格式规范对比表

| 方面 | 输入格式 | 输出格式 |
|------|----------|----------|
| **数据结构** | JSON对象 | Python列表/numpy数组 |
| **维度** | 11维网络监控数据 | 阶段1: [1,2], 阶段2: [1,6] |
| **数据类型** | float (JSON) | float32 (DLC输出) |
| **验证工具** | `simple_validate_json.py` | `process_dlc_output.py` |
| **示例文件** | `example_normal_input.json` | `example_dlc_outputs.json` |
| **处理复杂度** | 简单JSON解析 | 需要softmax后处理 |

---

## 🎯 输入输出字段映射

### 📥 输入字段（11维）
```python
输入字段序号                    字段名称                    数据范围
─────────────────────────────────────────────────────────────
index[0]  ← wlan0_wireless_quality     WiFi信号质量      [0, 100]
index[1]  ← wlan0_signal_level         WiFi信号强度      [-100, -10] dBm
index[2]  ← wlan0_noise_level          WiFi噪声水平      [-100, -30] dBm
index[3]  ← wlan0_rx_packets           接收包数          [0, +∞]
index[4]  ← wlan0_tx_packets           发送包数          [0, +∞]
index[5]  ← wlan0_rx_bytes             接收字节数        [0, +∞]
index[6]  ← wlan0_tx_bytes             发送字节数        [0, +∞]
index[7]  ← gateway_ping_time          网关ping时间      [0, 5000] ms
index[8]  ← dns_resolution_time        DNS解析时间       [0, 5000] ms
index[9]  ← memory_usage_percent       内存使用率        [0, 100] %
index[10] ← cpu_usage_percent          CPU使用率         [0, 100] %
```

### 📤 输出字段映射
```python
# 阶段1：异常检测输出 [1, 2]
[异常logit, 正常logit] → softmax → [异常概率, 正常概率]

# 阶段2：异常分类输出 [1, 6]
[wifi_degradation, network_latency, connection_instability,
 bandwidth_congestion, system_stress, dns_issues] → softmax → 6种异常类型概率
```

---

## 🚀 集成示例

### 📝 完整的输入输出处理示例
```python
import json
from simple_validate_json import validate_json_input, json_to_model_input_list
from process_dlc_output import DLCOutputProcessor

# 1. 输入处理
with open('example_normal_input.json', 'r') as f:
    input_data = json.load(f)

# 验证输入格式
is_valid, errors = validate_json_input(input_data)
if not is_valid:
    print(f"输入验证失败: {errors}")
    exit(1)

# 转换为模型输入
model_input = json_to_model_input_list(input_data)  # 11维列表

# 2. DLC模型推理（伪代码）
# detection_output = dlc_detector.predict(model_input)  # [1, 2]
# classification_output = dlc_classifier.predict(model_input)  # [1, 6]

# 模拟DLC输出
detection_output = [[-2.1543, 3.8967]]  # 正常样本
classification_output = None  # 正常样本不需要分类

# 3. 输出处理
processor = DLCOutputProcessor()
detection_result = processor.process_detection_output(detection_output)

if detection_result['is_anomaly']:
    classification_result = processor.process_classification_output(classification_output)
else:
    classification_result = None

# 4. 最终结果整合
final_result = processor.integrate_results(
    input_data=input_data,
    detection_result=detection_result,
    classification_result=classification_result,
    processing_time_ms=8.5
)

print(json.dumps(final_result, indent=2, ensure_ascii=False))
```

---

## 📚 相关资源

### 📖 技术文档
- **项目README**: `README.md` - 项目整体说明
- **DLC转换脚本**: `convert_realistic_end_to_end_to_dlc.py`
- **端到端测试**: `test_realistic_end_to_end_system.py`

### 🧪 测试工具
- **输入验证**: `python3 simple_validate_json.py`
- **输出处理**: `python3 process_dlc_output.py`
- **系统测试**: `python3 test_complete_system.py`

### 🎯 模型文件
- **异常检测DLC**: `realistic_end_to_end_anomaly_detector.dlc` (57.1 KB)
- **异常分类DLC**: `realistic_end_to_end_anomaly_classifier.dlc` (190.2 KB)
- **数据标准化器**: `realistic_raw_data_scaler.pkl` (0.8 KB)

---

## ⚠️ 重要注意事项

### ✅ 格式一致性
1. **输入输出格式必须严格匹配**：DLC转换前后的输入输出格式完全一致
2. **数据类型一致**：输入使用float，输出处理使用float32
3. **维度顺序固定**：11维输入和2维/6维输出的顺序不能改变

### 🔧 最佳实践
1. **始终进行输入验证**：使用提供的验证工具检查输入格式
2. **完整的输出处理**：包括softmax计算、置信度评估和结果整合
3. **错误处理机制**：对异常输入和输出提供恰当的错误处理
4. **文档同步更新**：修改格式时同步更新所有相关文档

### 📊 性能考虑
- **输入验证开销**：约1-2ms，建议在生产环境中启用
- **输出处理开销**：约0.5-1ms，主要是numpy计算
- **总处理时间**：完整流程约8-12ms，满足实时要求

---

**🎯 总结**: 本索引提供了DLC模型输入输出格式的完整指南。遵循这些规范可以确保系统的可靠性和一致性。如有疑问，请参考具体的规范文档或使用提供的验证工具。** 