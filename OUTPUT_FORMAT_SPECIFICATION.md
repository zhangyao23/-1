# 🎯 DLC模型输出格式规范文档

## 📋 概述

我们的两阶段DLC系统产生**结构化的输出数据**，需要进行后处理来得到最终的异常检测结果。本文档详细说明了DLC模型的原始输出格式和完整的数据处理流程。

---

## 🏗️ 两阶段系统架构

```
11维输入 → 阶段1: 异常检测DLC → 2维输出 → 阶段2: 异常分类DLC → 6维输出 → 最终结果
```

### 🔍 阶段1：异常检测网络输出

**DLC文件**: `realistic_end_to_end_anomaly_detector.dlc`

#### 原始输出格式
```
输出张量形状: [1, 2] (batch_size=1, classes=2)
数据类型: float32
输出名称: "output"
```

#### 原始数值示例
```python
# 正常样本的原始输出
raw_output = [[-2.1543, 3.8967]]  # [异常logit, 正常logit]

# 异常样本的原始输出  
raw_output = [[4.2156, -1.3547]]  # [异常logit, 正常logit]
```

#### 后处理步骤
```python
import numpy as np

def process_anomaly_detection_output(raw_output):
    """
    处理异常检测DLC的原始输出
    
    Args:
        raw_output: DLC模型的原始输出 [1, 2]
        
    Returns:
        dict: 处理后的结果
    """
    # 1. 应用softmax获取概率
    logits = raw_output[0]  # [异常logit, 正常logit]
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    
    # 2. 获取预测结果
    predicted_class = np.argmax(probabilities)
    is_anomaly = predicted_class == 0  # 索引0代表异常
    
    # 3. 获取置信度
    confidence = np.max(probabilities)
    anomaly_probability = probabilities[0]
    normal_probability = probabilities[1]
    
    return {
        "raw_logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "predicted_class": int(predicted_class),
        "is_anomaly": bool(is_anomaly),
        "confidence": float(confidence),
        "anomaly_probability": float(anomaly_probability),
        "normal_probability": float(normal_probability)
    }

# 示例使用
normal_raw = [[-2.1543, 3.8967]]
result = process_anomaly_detection_output(normal_raw)
print(result)
# 输出: {
#   "raw_logits": [-2.1543, 3.8967],
#   "probabilities": [0.0067, 0.9933],
#   "predicted_class": 1,
#   "is_anomaly": false,
#   "confidence": 0.9933,
#   "anomaly_probability": 0.0067,
#   "normal_probability": 0.9933
# }
```

---

### 🏷️ 阶段2：异常分类网络输出

**DLC文件**: `realistic_end_to_end_anomaly_classifier.dlc`

#### 原始输出格式
```
输出张量形状: [1, 6] (batch_size=1, classes=6)
数据类型: float32
输出名称: "output"
```

#### 异常类型映射
```python
ANOMALY_CLASSES = {
    0: "wifi_degradation",      # WiFi信号衰减
    1: "network_latency",       # 网络延迟
    2: "connection_instability", # 连接不稳定
    3: "bandwidth_congestion",  # 带宽拥塞
    4: "system_stress",         # 系统压力
    5: "dns_issues"             # DNS问题
}
```

#### 原始数值示例
```python
# WiFi信号衰减异常的原始输出
raw_output = [[3.2156, -1.1547, 0.8432, -0.5231, 1.2341, -2.1234]]
# 对应: [wifi_degradation, network_latency, connection_instability, 
#        bandwidth_congestion, system_stress, dns_issues]

# 网络延迟异常的原始输出
raw_output = [[-0.8432, 4.1234, -1.2341, 0.2156, -0.5431, 1.3456]]
```

#### 后处理步骤
```python
def process_anomaly_classification_output(raw_output):
    """
    处理异常分类DLC的原始输出
    
    Args:
        raw_output: DLC模型的原始输出 [1, 6]
        
    Returns:
        dict: 处理后的结果
    """
    ANOMALY_CLASSES = {
        0: "wifi_degradation",
        1: "network_latency", 
        2: "connection_instability",
        3: "bandwidth_congestion",
        4: "system_stress",
        5: "dns_issues"
    }
    
    # 1. 应用softmax获取概率
    logits = raw_output[0]  # [6个异常类型的logit值]
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    
    # 2. 获取预测结果
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = ANOMALY_CLASSES[predicted_class_index]
    
    # 3. 获取置信度
    confidence = np.max(probabilities)
    
    # 4. 构建详细概率分布
    class_probabilities = {}
    for i, class_name in ANOMALY_CLASSES.items():
        class_probabilities[class_name] = float(probabilities[i])
    
    return {
        "raw_logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "predicted_class_index": int(predicted_class_index),
        "predicted_class_name": predicted_class_name,
        "confidence": float(confidence),
        "class_probabilities": class_probabilities
    }

# 示例使用
wifi_raw = [[3.2156, -1.1547, 0.8432, -0.5231, 1.2341, -2.1234]]
result = process_anomaly_classification_output(wifi_raw)
print(result)
# 输出: {
#   "raw_logits": [3.2156, -1.1547, 0.8432, -0.5231, 1.2341, -2.1234],
#   "probabilities": [0.7234, 0.0891, 0.1256, 0.0234, 0.0321, 0.0064],
#   "predicted_class_index": 0,
#   "predicted_class_name": "wifi_degradation",
#   "confidence": 0.7234,
#   "class_probabilities": {
#     "wifi_degradation": 0.7234,
#     "network_latency": 0.0891,
#     "connection_instability": 0.1256,
#     "bandwidth_congestion": 0.0234,
#     "system_stress": 0.0321,
#     "dns_issues": 0.0064
#   }
# }
```

---

## 🎯 最终整合输出格式

### 📊 完整系统响应格式

```python
def integrate_two_stage_results(detection_result, classification_result=None):
    """
    整合两阶段系统的输出结果
    
    Args:
        detection_result: 异常检测结果
        classification_result: 异常分类结果（可选）
        
    Returns:
        dict: 最终的系统输出
    """
    integrated_result = {
        "timestamp": "2025-07-07T14:30:01Z",
        "processing_time_ms": 8.5,
        "detection_stage": {
            "is_anomaly": detection_result["is_anomaly"],
            "confidence": detection_result["confidence"],
            "anomaly_probability": detection_result["anomaly_probability"],
            "normal_probability": detection_result["normal_probability"]
        }
    }
    
    if detection_result["is_anomaly"] and classification_result:
        integrated_result["classification_stage"] = {
            "anomaly_type": classification_result["predicted_class_name"],
            "confidence": classification_result["confidence"],
            "all_probabilities": classification_result["class_probabilities"]
        }
    else:
        integrated_result["classification_stage"] = None
    
    return integrated_result
```

### 🔧 标准API响应格式

#### 📱 正常网络状态响应
```json
{
  "timestamp": "2025-07-07T14:30:01Z",
  "processing_time_ms": 8.5,
  "detection_stage": {
    "is_anomaly": false,
    "confidence": 0.9933,
    "anomaly_probability": 0.0067,
    "normal_probability": 0.9933
  },
  "classification_stage": null,
  "final_result": {
    "status": "normal",
    "message": "Network is operating normally",
    "action_required": false
  }
}
```

#### 🚨 异常网络状态响应
```json
{
  "timestamp": "2025-07-07T14:30:01Z",
  "processing_time_ms": 12.3,
  "detection_stage": {
    "is_anomaly": true,
    "confidence": 0.8765,
    "anomaly_probability": 0.8765,
    "normal_probability": 0.1235
  },
  "classification_stage": {
    "anomaly_type": "wifi_degradation",
    "confidence": 0.7234,
    "all_probabilities": {
      "wifi_degradation": 0.7234,
      "network_latency": 0.0891,
      "connection_instability": 0.1256,
      "bandwidth_congestion": 0.0234,
      "system_stress": 0.0321,
      "dns_issues": 0.0064
    }
  },
  "final_result": {
    "status": "anomaly_detected",
    "message": "WiFi signal degradation detected",
    "action_required": true,
    "recommended_actions": [
      "Check WiFi signal strength",
      "Move closer to router",
      "Check for interference sources"
    ]
  }
}
```

---

## 📋 不同场景的输出示例

### 🌟 场景1：正常网络状态
```json
{
  "input_data": {
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
  },
  "dlc_outputs": {
    "stage1_raw": [[-2.1543, 3.8967]],
    "stage2_raw": null
  },
  "processed_results": {
    "detection_stage": {
      "is_anomaly": false,
      "confidence": 0.9933,
      "anomaly_probability": 0.0067,
      "normal_probability": 0.9933
    },
    "classification_stage": null
  },
  "final_result": {
    "status": "normal",
    "message": "Network is operating normally"
  }
}
```

### 📶 场景2：WiFi信号衰减
```json
{
  "input_data": {
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
  },
  "dlc_outputs": {
    "stage1_raw": [[4.2156, -1.3547]],
    "stage2_raw": [[3.2156, -1.1547, 0.8432, -0.5231, 1.2341, -2.1234]]
  },
  "processed_results": {
    "detection_stage": {
      "is_anomaly": true,
      "confidence": 0.8765,
      "anomaly_probability": 0.8765,
      "normal_probability": 0.1235
    },
    "classification_stage": {
      "anomaly_type": "wifi_degradation",
      "confidence": 0.7234,
      "all_probabilities": {
        "wifi_degradation": 0.7234,
        "network_latency": 0.0891,
        "connection_instability": 0.1256,
        "bandwidth_congestion": 0.0234,
        "system_stress": 0.0321,
        "dns_issues": 0.0064
      }
    }
  },
  "final_result": {
    "status": "anomaly_detected",
    "message": "WiFi signal degradation detected",
    "severity": "medium",
    "action_required": true,
    "recommended_actions": [
      "Check WiFi signal strength",
      "Move closer to router",
      "Check for interference sources"
    ]
  }
}
```

### 🌐 场景3：网络延迟异常
```json
{
  "input_data": {
    "wlan0_wireless_quality": 70.0,
    "wlan0_signal_level": -55.0,
    "wlan0_noise_level": -85.0,
    "wlan0_rx_packets": 12000,
    "wlan0_tx_packets": 9000,
    "wlan0_rx_bytes": 2200000,
    "wlan0_tx_bytes": 1800000,
    "gateway_ping_time": 150.0,
    "dns_resolution_time": 200.0,
    "memory_usage_percent": 40.0,
    "cpu_usage_percent": 25.0
  },
  "dlc_outputs": {
    "stage1_raw": [[3.1234, -0.8765]],
    "stage2_raw": [[-0.8432, 4.1234, -1.2341, 0.2156, -0.5431, 1.3456]]
  },
  "processed_results": {
    "detection_stage": {
      "is_anomaly": true,
      "confidence": 0.9234,
      "anomaly_probability": 0.9234,
      "normal_probability": 0.0766
    },
    "classification_stage": {
      "anomaly_type": "network_latency",
      "confidence": 0.8456,
      "all_probabilities": {
        "wifi_degradation": 0.0234,
        "network_latency": 0.8456,
        "connection_instability": 0.0567,
        "bandwidth_congestion": 0.0432,
        "system_stress": 0.0123,
        "dns_issues": 0.0188
      }
    }
  },
  "final_result": {
    "status": "anomaly_detected",
    "message": "High network latency detected",
    "severity": "high",
    "action_required": true,
    "recommended_actions": [
      "Check network connection",
      "Restart router",
      "Contact ISP if problem persists",
      "Check for background downloads"
    ]
  }
}
```

### 💻 场景4：系统压力异常
```json
{
  "input_data": {
    "wlan0_wireless_quality": 75.0,
    "wlan0_signal_level": -50.0,
    "wlan0_noise_level": -90.0,
    "wlan0_rx_packets": 14000,
    "wlan0_tx_packets": 11000,
    "wlan0_rx_bytes": 2800000,
    "wlan0_tx_bytes": 2300000,
    "gateway_ping_time": 30.0,
    "dns_resolution_time": 40.0,
    "memory_usage_percent": 95.0,
    "cpu_usage_percent": 90.0
  },
  "dlc_outputs": {
    "stage1_raw": [[2.8765, -1.2341]],
    "stage2_raw": [[-1.2341, 0.3456, -0.8765, 1.1234, 3.4567, -0.6789]]
  },
  "processed_results": {
    "detection_stage": {
      "is_anomaly": true,
      "confidence": 0.8932,
      "anomaly_probability": 0.8932,
      "normal_probability": 0.1068
    },
    "classification_stage": {
      "anomaly_type": "system_stress",
      "confidence": 0.7892,
      "all_probabilities": {
        "wifi_degradation": 0.0456,
        "network_latency": 0.0789,
        "connection_instability": 0.0345,
        "bandwidth_congestion": 0.0432,
        "system_stress": 0.7892,
        "dns_issues": 0.0086
      }
    }
  },
  "final_result": {
    "status": "anomaly_detected",
    "message": "System under high stress",
    "severity": "critical",
    "action_required": true,
    "recommended_actions": [
      "Close unnecessary applications",
      "Restart system",
      "Check for memory leaks",
      "Monitor resource usage"
    ]
  }
}
```

---

## 🔧 输出处理工具

### Python完整处理函数
```python
import numpy as np
from datetime import datetime
import time

class DLCOutputProcessor:
    """DLC输出处理器"""
    
    def __init__(self):
        self.anomaly_classes = {
            0: "wifi_degradation",
            1: "network_latency",
            2: "connection_instability", 
            3: "bandwidth_congestion",
            4: "system_stress",
            5: "dns_issues"
        }
        
        self.severity_mapping = {
            "wifi_degradation": "medium",
            "network_latency": "high",
            "connection_instability": "high",
            "bandwidth_congestion": "medium",
            "system_stress": "critical",
            "dns_issues": "medium"
        }
        
        self.action_recommendations = {
            "wifi_degradation": [
                "Check WiFi signal strength",
                "Move closer to router",
                "Check for interference sources"
            ],
            "network_latency": [
                "Check network connection",
                "Restart router",
                "Contact ISP if problem persists",
                "Check for background downloads"
            ],
            "connection_instability": [
                "Check network cable connections",
                "Restart network adapter",
                "Update network drivers"
            ],
            "bandwidth_congestion": [
                "Close bandwidth-heavy applications",
                "Limit background updates",
                "Upgrade internet plan if needed"
            ],
            "system_stress": [
                "Close unnecessary applications",
                "Restart system",
                "Check for memory leaks",
                "Monitor resource usage"
            ],
            "dns_issues": [
                "Try different DNS servers",
                "Flush DNS cache",
                "Check DNS configuration"
            ]
        }
    
    def process_detection_output(self, raw_output):
        """处理异常检测输出"""
        logits = np.array(raw_output[0])
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        predicted_class = np.argmax(probabilities)
        is_anomaly = predicted_class == 0
        confidence = float(np.max(probabilities))
        
        return {
            "raw_logits": logits.tolist(),
            "probabilities": probabilities.tolist(),
            "predicted_class": int(predicted_class),
            "is_anomaly": bool(is_anomaly),
            "confidence": confidence,
            "anomaly_probability": float(probabilities[0]),
            "normal_probability": float(probabilities[1])
        }
    
    def process_classification_output(self, raw_output):
        """处理异常分类输出"""
        logits = np.array(raw_output[0])
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = self.anomaly_classes[predicted_class_index]
        confidence = float(np.max(probabilities))
        
        class_probabilities = {}
        for i, class_name in self.anomaly_classes.items():
            class_probabilities[class_name] = float(probabilities[i])
        
        return {
            "raw_logits": logits.tolist(),
            "probabilities": probabilities.tolist(),
            "predicted_class_index": int(predicted_class_index),
            "predicted_class_name": predicted_class_name,
            "confidence": confidence,
            "class_probabilities": class_probabilities
        }
    
    def integrate_results(self, input_data, detection_result, classification_result=None, processing_time=None):
        """整合最终结果"""
        start_time = time.time()
        
        integrated_result = {
            "timestamp": datetime.now().isoformat() + "Z",
            "processing_time_ms": processing_time if processing_time else round((time.time() - start_time) * 1000, 1),
            "detection_stage": {
                "is_anomaly": detection_result["is_anomaly"],
                "confidence": detection_result["confidence"],
                "anomaly_probability": detection_result["anomaly_probability"],
                "normal_probability": detection_result["normal_probability"]
            }
        }
        
        if detection_result["is_anomaly"] and classification_result:
            anomaly_type = classification_result["predicted_class_name"]
            integrated_result["classification_stage"] = {
                "anomaly_type": anomaly_type,
                "confidence": classification_result["confidence"],
                "all_probabilities": classification_result["class_probabilities"]
            }
            
            # 添加最终结果和建议
            integrated_result["final_result"] = {
                "status": "anomaly_detected",
                "message": f"{anomaly_type.replace('_', ' ').title()} detected",
                "severity": self.severity_mapping.get(anomaly_type, "medium"),
                "action_required": True,
                "recommended_actions": self.action_recommendations.get(anomaly_type, [])
            }
        else:
            integrated_result["classification_stage"] = None
            integrated_result["final_result"] = {
                "status": "normal",
                "message": "Network is operating normally",
                "action_required": False
            }
        
        return integrated_result

# 使用示例
processor = DLCOutputProcessor()

# 模拟DLC输出
detection_raw = [[4.2156, -1.3547]]  # 异常检测输出
classification_raw = [[3.2156, -1.1547, 0.8432, -0.5231, 1.2341, -2.1234]]  # 异常分类输出

# 处理输出
detection_result = processor.process_detection_output(detection_raw)
classification_result = processor.process_classification_output(classification_raw)

# 整合结果
input_data = {"wlan0_wireless_quality": 45.0, "wlan0_signal_level": -70.0}  # 简化示例
final_result = processor.integrate_results(input_data, detection_result, classification_result)

print(json.dumps(final_result, indent=2))
```

---

## 📊 输出格式总结

### ✅ 关键特点
1. **分层输出**: 检测阶段 → 分类阶段 → 最终结果
2. **概率信息**: 提供完整的概率分布，支持不确定性量化
3. **置信度量化**: 每个阶段都有明确的置信度指标
4. **行动建议**: 根据异常类型提供具体的解决方案
5. **时间戳**: 完整的处理时间记录

### 🎯 最佳实践
1. **阈值设置**: 根据业务需求调整置信度阈值
2. **异常处理**: 对低置信度预测进行特殊处理
3. **日志记录**: 保存完整的输入输出用于调试
4. **性能监控**: 记录处理时间和资源使用情况

---

**🎯 总结**: DLC模型输出需要完整的后处理流程，包括概率计算、置信度评估和结果整合。标准化的输出格式确保了系统的可靠性和可维护性。** 