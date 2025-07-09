# 🎯 模型输入格式规范文档

## 📋 概述

我们的DLC模型要求**严格的11维浮点数输入**，数据必须按照**固定顺序**排列。为了标准化输入，我们定义了标准的JSON输入格式。

---

## 📊 标准JSON输入格式

### 🎯 完整JSON模板
```json
{
  "timestamp": "2025-07-07T14:30:00Z",
  "device_id": "device_001",
  "network_data": {
    "wlan0_wireless_quality": 75.0,
    "wlan0_signal_level": -50.0,
    "wlan0_noise_level": -90.0,
    "wlan0_rx_packets": 15000,
    "wlan0_tx_packets": 12000,
    "wlan0_rx_bytes": 3000000,
    "wlan0_tx_bytes": 2500000,
    "gateway_ping_time": 20.0,
    "dns_resolution_time": 30.0,
    "memory_usage_percent": 40.0,
    "cpu_usage_percent": 25.0
  }
}
```

### 🔢 模型输入向量映射
```
JSON字段 → 模型输入位置 → 数据类型 → 取值范围
─────────────────────────────────────────────────────
network_data.wlan0_wireless_quality    → index[0]  → float32 → [0, 100]
network_data.wlan0_signal_level         → index[1]  → float32 → [-100, -10] (dBm)
network_data.wlan0_noise_level          → index[2]  → float32 → [-100, -30] (dBm)
network_data.wlan0_rx_packets           → index[3]  → float32 → [0, +∞]
network_data.wlan0_tx_packets           → index[4]  → float32 → [0, +∞]
network_data.wlan0_rx_bytes             → index[5]  → float32 → [0, +∞]
network_data.wlan0_tx_bytes             → index[6]  → float32 → [0, +∞]
network_data.gateway_ping_time          → index[7]  → float32 → [0, 5000] (ms)
network_data.dns_resolution_time        → index[8]  → float32 → [0, 5000] (ms)
network_data.memory_usage_percent       → index[9]  → float32 → [0, 100] (%)
network_data.cpu_usage_percent          → index[10] → float32 → [0, 100] (%)
```

---

## 🔄 JSON到模型输入的转换

### Python转换示例
```python
import json
import numpy as np

def json_to_model_input(json_data):
    """
    将JSON格式的网络数据转换为模型输入向量
    
    Args:
        json_data: JSON字符串或dict对象
        
    Returns:
        numpy.ndarray: 11维float32向量
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    network_data = data["network_data"]
    
    # 按照固定顺序提取数据
    input_vector = np.array([
        float(network_data["wlan0_wireless_quality"]),
        float(network_data["wlan0_signal_level"]),
        float(network_data["wlan0_noise_level"]),
        float(network_data["wlan0_rx_packets"]),
        float(network_data["wlan0_tx_packets"]),
        float(network_data["wlan0_rx_bytes"]),
        float(network_data["wlan0_tx_bytes"]),
        float(network_data["gateway_ping_time"]),
        float(network_data["dns_resolution_time"]),
        float(network_data["memory_usage_percent"]),
        float(network_data["cpu_usage_percent"])
    ], dtype=np.float32)
    
    return input_vector

# 使用示例
json_input = """{
  "timestamp": "2025-07-07T14:30:00Z",
  "device_id": "device_001",
  "network_data": {
    "wlan0_wireless_quality": 75.0,
    "wlan0_signal_level": -50.0,
    "wlan0_noise_level": -90.0,
    "wlan0_rx_packets": 15000,
    "wlan0_tx_packets": 12000,
    "wlan0_rx_bytes": 3000000,
    "wlan0_tx_bytes": 2500000,
    "gateway_ping_time": 20.0,
    "dns_resolution_time": 30.0,
    "memory_usage_percent": 40.0,
    "cpu_usage_percent": 25.0
  }
}"""

model_input = json_to_model_input(json_input)
print(f"Model input shape: {model_input.shape}")
print(f"Model input: {model_input}")
```

### C++转换示例
```cpp
#include <nlohmann/json.hpp>
#include <vector>

std::vector<float> jsonToModelInput(const std::string& json_str) {
    nlohmann::json data = nlohmann::json::parse(json_str);
    auto network_data = data["network_data"];
    
    std::vector<float> input_vector = {
        network_data["wlan0_wireless_quality"].get<float>(),
        network_data["wlan0_signal_level"].get<float>(),
        network_data["wlan0_noise_level"].get<float>(),
        network_data["wlan0_rx_packets"].get<float>(),
        network_data["wlan0_tx_packets"].get<float>(),
        network_data["wlan0_rx_bytes"].get<float>(),
        network_data["wlan0_tx_bytes"].get<float>(),
        network_data["gateway_ping_time"].get<float>(),
        network_data["dns_resolution_time"].get<float>(),
        network_data["memory_usage_percent"].get<float>(),
        network_data["cpu_usage_percent"].get<float>()
    };
    
    return input_vector;
}
```

---

## ⚠️ 数据验证规则

### 🔍 必需字段检查
```python
def validate_json_input(data):
    """验证JSON输入的完整性和有效性"""
    required_fields = [
        "wlan0_wireless_quality",
        "wlan0_signal_level", 
        "wlan0_noise_level",
        "wlan0_rx_packets",
        "wlan0_tx_packets",
        "wlan0_rx_bytes",
        "wlan0_tx_bytes",
        "gateway_ping_time",
        "dns_resolution_time",
        "memory_usage_percent",
        "cpu_usage_percent"
    ]
    
    network_data = data.get("network_data", {})
    
    # 检查必需字段
    missing_fields = []
    for field in required_fields:
        if field not in network_data:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # 检查数据类型
    for field in required_fields:
        value = network_data[field]
        if not isinstance(value, (int, float)):
            raise ValueError(f"Field {field} must be numeric, got {type(value)}")
    
    # 检查数值范围
    validation_rules = {
        "wlan0_wireless_quality": (0, 100),
        "wlan0_signal_level": (-100, -10),
        "wlan0_noise_level": (-100, -30),
        "wlan0_rx_packets": (0, float('inf')),
        "wlan0_tx_packets": (0, float('inf')),
        "wlan0_rx_bytes": (0, float('inf')),
        "wlan0_tx_bytes": (0, float('inf')),
        "gateway_ping_time": (0, 5000),
        "dns_resolution_time": (0, 5000),
        "memory_usage_percent": (0, 100),
        "cpu_usage_percent": (0, 100)
    }
    
    for field, (min_val, max_val) in validation_rules.items():
        value = network_data[field]
        if not (min_val <= value <= max_val):
            raise ValueError(f"Field {field} value {value} out of range [{min_val}, {max_val}]")
    
    return True
```

---

## 🎯 不同场景的输入示例

### 📱 正常网络状态
```json
{
  "timestamp": "2025-07-07T14:30:00Z",
  "device_id": "device_001",
  "network_data": {
    "wlan0_wireless_quality": 85.0,
    "wlan0_signal_level": -45.0,
    "wlan0_noise_level": -92.0,
    "wlan0_rx_packets": 18500,
    "wlan0_tx_packets": 15200,
    "wlan0_rx_bytes": 3500000,
    "wlan0_tx_bytes": 2800000,
    "gateway_ping_time": 15.0,
    "dns_resolution_time": 25.0,
    "memory_usage_percent": 35.0,
    "cpu_usage_percent": 20.0
  }
}
```

### 📶 WiFi信号衰减异常
```json
{
  "timestamp": "2025-07-07T14:31:00Z",
  "device_id": "device_001", 
  "network_data": {
    "wlan0_wireless_quality": 45.0,
    "wlan0_signal_level": -70.0,
    "wlan0_noise_level": -75.0,
    "wlan0_rx_packets": 6000,
    "wlan0_tx_packets": 4500,
    "wlan0_rx_bytes": 1200000,
    "wlan0_tx_bytes": 1000000,
    "gateway_ping_time": 45.0,
    "dns_resolution_time": 60.0,
    "memory_usage_percent": 45.0,
    "cpu_usage_percent": 30.0
  }
}
```

### 🌐 网络延迟异常
```json
{
  "timestamp": "2025-07-07T14:32:00Z",
  "device_id": "device_001",
  "network_data": {
    "wlan0_wireless_quality": 70.0,
    "wlan0_signal_level": -55.0,
    "wlan0_noise_level": -85.0,
    "wlan0_rx_packets": 12000,
    "wlan0_tx_packets": 9000,
    "wlan0_rx_bytes": 2200000,
    "wlan0_tx_bytes": 1800000,
    "gateway_ping_time": 80.0,
    "dns_resolution_time": 120.0,
    "memory_usage_percent": 40.0,
    "cpu_usage_percent": 25.0
  }
}
```

### 💻 系统压力异常
```json
{
  "timestamp": "2025-07-07T14:33:00Z",
  "device_id": "device_001",
  "network_data": {
    "wlan0_wireless_quality": 75.0,
    "wlan0_signal_level": -50.0,
    "wlan0_noise_level": -90.0,
    "wlan0_rx_packets": 14000,
    "wlan0_tx_packets": 11000,
    "wlan0_rx_bytes": 2800000,
    "wlan0_tx_bytes": 2300000,
    "gateway_ping_time": 30.0,
    "dns_resolution_time": 40.0,
    "memory_usage_percent": 85.0,
    "cpu_usage_percent": 80.0
  }
}
```

---

## 🔧 API接口输入格式

### POST /api/v1/predict
```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-07-07T14:30:00Z",
    "device_id": "device_001",
    "network_data": {
      "wlan0_wireless_quality": 75.0,
      "wlan0_signal_level": -50.0,
      "wlan0_noise_level": -90.0,
      "wlan0_rx_packets": 15000,
      "wlan0_tx_packets": 12000,
      "wlan0_rx_bytes": 3000000,
      "wlan0_tx_bytes": 2500000,
      "gateway_ping_time": 20.0,
      "dns_resolution_time": 30.0,
      "memory_usage_percent": 40.0,
      "cpu_usage_percent": 25.0
    }
  }'
```

### 预期响应格式
```json
{
  "is_anomaly": false,
  "anomaly_type": null,
  "detection_confidence": 0.999,
  "classification_confidence": null,
  "timestamp": "2025-07-07T14:30:01Z",
  "processing_time_ms": 8.5
}
```

---

## 📝 输入格式总结

### ✅ 关键要求
1. **固定顺序**: 11个字段必须按照指定顺序
2. **数据类型**: 所有数值必须是float32类型
3. **完整性**: 不能缺少任何字段
4. **范围校验**: 每个字段都有合理的取值范围
5. **时间戳**: 建议包含时间戳用于追踪

### 🎯 最佳实践
1. **数据校验**: 输入前进行完整性和范围检查
2. **异常处理**: 对无效输入提供清晰的错误信息
3. **类型转换**: 确保数值类型正确转换为float32
4. **文档同步**: 保持输入格式文档与代码同步更新

---

## 📋 相关文档

### 🔗 输出格式规范
- **输出格式详细说明**: `OUTPUT_FORMAT_SPECIFICATION.md`
- **输出处理工具**: `process_dlc_output.py`
- **输出示例数据**: `example_dlc_outputs.json`

### 🛠️ 工具使用
```bash
# 验证输入格式
python3 simple_validate_json.py example_normal_input.json

# 处理输出格式
python3 process_dlc_output.py

# 查看输出示例
cat example_dlc_outputs.json
```

### 🎯 完整流程
```
输入验证 → DLC推理 → 输出处理 → 最终结果
     ↓           ↓          ↓         ↓
INPUT_FORMAT → DLC文件 → OUTPUT_FORMAT → JSON响应
```

---

**🎯 总结**: 模型输入必须严格遵循11维float32向量格式，JSON只是数据传输的标准化格式。关键是要确保JSON到模型输入的转换过程正确且一致。输出格式同样重要，请参考相关文档进行完整的输入输出处理。** 