#!/usr/bin/env python3
"""
JSON输入格式验证工具
验证网络监控数据JSON是否符合模型输入要求
"""

import json
import numpy as np
import sys
from typing import Dict, Any, Tuple, List

def validate_json_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证JSON输入的完整性和有效性
    
    Args:
        data: 解析后的JSON数据
        
    Returns:
        (is_valid, error_messages): 验证结果和错误信息列表
    """
    errors = []
    
    # 检查顶层结构
    if "network_data" not in data:
        errors.append("Missing 'network_data' field")
        return False, errors
    
    network_data = data["network_data"]
    
    # 必需字段列表
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
    
    # 检查必需字段
    missing_fields = []
    for field in required_fields:
        if field not in network_data:
            missing_fields.append(field)
    
    if missing_fields:
        errors.append(f"Missing required fields: {missing_fields}")
    
    # 检查数据类型
    for field in required_fields:
        if field in network_data:
            value = network_data[field]
            if not isinstance(value, (int, float)):
                errors.append(f"Field '{field}' must be numeric, got {type(value).__name__}: {value}")
    
    # 检查数值范围
    validation_rules = {
        "wlan0_wireless_quality": (0, 100, "WiFi信号质量百分比"),
        "wlan0_signal_level": (-100, -10, "WiFi信号强度(dBm)"),
        "wlan0_noise_level": (-100, -30, "WiFi噪声水平(dBm)"),
        "wlan0_rx_packets": (0, float('inf'), "接收包数"),
        "wlan0_tx_packets": (0, float('inf'), "发送包数"),
        "wlan0_rx_bytes": (0, float('inf'), "接收字节数"),
        "wlan0_tx_bytes": (0, float('inf'), "发送字节数"),
        "gateway_ping_time": (0, 5000, "网关ping时间(ms)"),
        "dns_resolution_time": (0, 5000, "DNS解析时间(ms)"),
        "memory_usage_percent": (0, 100, "内存使用率(%)"),
        "cpu_usage_percent": (0, 100, "CPU使用率(%)")
    }
    
    for field, (min_val, max_val, description) in validation_rules.items():
        if field in network_data:
            value = network_data[field]
            if isinstance(value, (int, float)):
                if not (min_val <= value <= max_val):
                    errors.append(f"Field '{field}' ({description}) value {value} out of range [{min_val}, {max_val}]")
    
    return len(errors) == 0, errors

def json_to_model_input(data: Dict[str, Any]) -> np.ndarray:
    """
    将JSON格式的网络数据转换为模型输入向量
    
    Args:
        data: 解析后的JSON数据
        
    Returns:
        numpy.ndarray: 11维float32向量
    """
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

def test_with_examples():
    """测试几个示例输入"""
    print("🧪 **测试标准输入示例**\n")
    
    # 正常网络状态示例
    normal_example = {
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
    
    print("📱 **正常网络状态测试**:")
    is_valid, errors = validate_json_input(normal_example)
    if is_valid:
        print("✅ 验证通过")
        input_vector = json_to_model_input(normal_example)
        print(f"   模型输入向量形状: {input_vector.shape}")
        print(f"   模型输入向量: {input_vector}")
    else:
        print("❌ 验证失败:")
        for error in errors:
            print(f"   - {error}")
    print()
    
    # WiFi异常示例
    wifi_anomaly_example = {
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
    
    print("📶 **WiFi信号衰减异常测试**:")
    is_valid, errors = validate_json_input(wifi_anomaly_example)
    if is_valid:
        print("✅ 验证通过")
        input_vector = json_to_model_input(wifi_anomaly_example)
        print(f"   模型输入向量: {input_vector}")
    else:
        print("❌ 验证失败:")
        for error in errors:
            print(f"   - {error}")
    print()
    
    # 错误示例 - 缺少字段
    invalid_example = {
        "timestamp": "2025-07-07T14:32:00Z",
        "device_id": "device_001",
        "network_data": {
            "wlan0_wireless_quality": 75.0,
            "wlan0_signal_level": -50.0,
            # 缺少其他字段
        }
    }
    
    print("❌ **错误输入测试（缺少字段）**:")
    is_valid, errors = validate_json_input(invalid_example)
    if is_valid:
        print("✅ 验证通过")
    else:
        print("❌ 验证失败（预期结果）:")
        for error in errors:
            print(f"   - {error}")

def validate_file(filename: str):
    """验证JSON文件"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📁 **验证文件**: {filename}")
        is_valid, errors = validate_json_input(data)
        
        if is_valid:
            print("✅ 文件验证通过")
            input_vector = json_to_model_input(data)
            print(f"   模型输入向量形状: {input_vector.shape}")
            print(f"   模型输入向量: {input_vector}")
            return True
        else:
            print("❌ 文件验证失败:")
            for error in errors:
                print(f"   - {error}")
            return False
            
    except FileNotFoundError:
        print(f"❌ 文件未找到: {filename}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        return False

def validate_string(json_string: str):
    """验证JSON字符串"""
    try:
        data = json.loads(json_string)
        
        print("📝 **验证JSON字符串**:")
        is_valid, errors = validate_json_input(data)
        
        if is_valid:
            print("✅ 字符串验证通过")
            input_vector = json_to_model_input(data)
            print(f"   模型输入向量形状: {input_vector.shape}")
            print(f"   模型输入向量: {input_vector}")
            return True
        else:
            print("❌ 字符串验证失败:")
            for error in errors:
                print(f"   - {error}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        return False

def main():
    """主函数"""
    print("🎯 **JSON输入格式验证工具**")
    print("验证网络监控数据JSON是否符合DLC模型输入要求")
    print("=" * 60)
    print()
    
    if len(sys.argv) > 1:
        # 验证命令行提供的文件
        filename = sys.argv[1]
        validate_file(filename)
    else:
        # 运行内置示例测试
        test_with_examples()
        
        print("\n💡 **使用方法**:")
        print(f"   验证JSON文件: python {sys.argv[0]} your_input.json")
        print(f"   运行示例测试: python {sys.argv[0]}")

if __name__ == "__main__":
    main() 