#!/usr/bin/env python3
"""
简化版JSON输入格式验证工具
验证网络监控数据JSON是否符合模型输入要求（不依赖外部库）
"""

import json
import sys

def validate_json_input(data):
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
    
    # 必需字段列表（严格按照模型输入顺序）
    required_fields = [
        "wlan0_wireless_quality",      # index[0]
        "wlan0_signal_level",          # index[1] 
        "wlan0_noise_level",           # index[2]
        "wlan0_rx_packets",            # index[3]
        "wlan0_tx_packets",            # index[4]
        "wlan0_rx_bytes",              # index[5]
        "wlan0_tx_bytes",              # index[6]
        "gateway_ping_time",           # index[7]
        "dns_resolution_time",         # index[8]
        "memory_usage_percent",        # index[9]
        "cpu_usage_percent"            # index[10]
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

def json_to_model_input_list(data):
    """
    将JSON格式的网络数据转换为模型输入列表
    
    Args:
        data: 解析后的JSON数据
        
    Returns:
        list: 11维float列表（严格按照模型要求的顺序）
    """
    network_data = data["network_data"]
    
    # 按照固定顺序提取数据（严格按照模型输入索引顺序）
    input_list = [
        float(network_data["wlan0_wireless_quality"]),    # index[0]
        float(network_data["wlan0_signal_level"]),        # index[1]
        float(network_data["wlan0_noise_level"]),         # index[2]
        float(network_data["wlan0_rx_packets"]),          # index[3]
        float(network_data["wlan0_tx_packets"]),          # index[4]
        float(network_data["wlan0_rx_bytes"]),            # index[5]
        float(network_data["wlan0_tx_bytes"]),            # index[6]
        float(network_data["gateway_ping_time"]),         # index[7]
        float(network_data["dns_resolution_time"]),       # index[8]
        float(network_data["memory_usage_percent"]),      # index[9]
        float(network_data["cpu_usage_percent"])          # index[10]
    ]
    
    return input_list

def display_model_input_mapping(data):
    """显示JSON字段到模型输入的映射关系"""
    network_data = data["network_data"]
    
    print("🔢 **JSON字段 → 模型输入映射**:")
    print("=" * 70)
    
    field_mapping = [
        ("wlan0_wireless_quality", 0, "WiFi信号质量百分比"),
        ("wlan0_signal_level", 1, "WiFi信号强度(dBm)"),
        ("wlan0_noise_level", 2, "WiFi噪声水平(dBm)"),
        ("wlan0_rx_packets", 3, "接收包数"),
        ("wlan0_tx_packets", 4, "发送包数"),
        ("wlan0_rx_bytes", 5, "接收字节数"),
        ("wlan0_tx_bytes", 6, "发送字节数"),
        ("gateway_ping_time", 7, "网关ping时间(ms)"),
        ("dns_resolution_time", 8, "DNS解析时间(ms)"),
        ("memory_usage_percent", 9, "内存使用率(%)"),
        ("cpu_usage_percent", 10, "CPU使用率(%)")
    ]
    
    for field, index, description in field_mapping:
        value = network_data.get(field, "MISSING")
        print(f"index[{index:2d}] ← {field:25} = {value:>8} ({description})")

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
        input_list = json_to_model_input_list(normal_example)
        print(f"   模型输入维度: {len(input_list)}")
        print(f"   模型输入向量: {input_list}")
        print()
        display_model_input_mapping(normal_example)
    else:
        print("❌ 验证失败:")
        for error in errors:
            print(f"   - {error}")
    print("\n" + "=" * 70 + "\n")
    
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
        input_list = json_to_model_input_list(wifi_anomaly_example)
        print(f"   模型输入向量: {input_list}")
    else:
        print("❌ 验证失败:")
        for error in errors:
            print(f"   - {error}")
    print("\n" + "=" * 70 + "\n")
    
    # 错误示例 - 缺少字段
    invalid_example = {
        "timestamp": "2025-07-07T14:32:00Z",
        "device_id": "device_001",
        "network_data": {
            "wlan0_wireless_quality": 75.0,
            "wlan0_signal_level": -50.0,
            # 缺少其他9个字段
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

def validate_file(filename):
    """验证JSON文件"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📁 **验证文件**: {filename}")
        print("=" * 70)
        is_valid, errors = validate_json_input(data)
        
        if is_valid:
            print("✅ 文件验证通过")
            input_list = json_to_model_input_list(data)
            print(f"   模型输入维度: {len(input_list)}")
            print(f"   模型输入向量: {input_list}")
            print()
            display_model_input_mapping(data)
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

def main():
    """主函数"""
    print("🎯 **JSON输入格式验证工具**")
    print("验证网络监控数据JSON是否符合DLC模型输入要求")
    print("=" * 70)
    print()
    
    if len(sys.argv) > 1:
        # 验证命令行提供的文件
        filename = sys.argv[1]
        validate_file(filename)
    else:
        # 运行内置示例测试
        test_with_examples()
        
        print("\n💡 **使用方法**:")
        print(f"   验证JSON文件: python3 {sys.argv[0]} your_input.json")
        print(f"   运行示例测试: python3 {sys.argv[0]}")

if __name__ == "__main__":
    main() 