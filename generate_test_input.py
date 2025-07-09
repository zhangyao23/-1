#!/usr/bin/env python3
"""
测试输入数据生成工具
为移动设备DLC推理生成二进制格式的测试数据
"""

import struct
import json
import sys
import os
from typing import List, Dict, Any, Optional

def create_test_scenarios() -> Dict[str, Dict[str, Any]]:
    """创建测试场景数据"""
    
    scenarios = {
        "normal": {
            "name": "正常网络状态",
            "description": "所有网络指标都在正常范围内",
            "data": {
                "wlan0_wireless_quality": 85.0,
                "wlan0_signal_level": -45.0,
                "wlan0_noise_level": -92.0,
                "wlan0_rx_packets": 18500.0,
                "wlan0_tx_packets": 15200.0,
                "wlan0_rx_bytes": 3500000.0,
                "wlan0_tx_bytes": 2800000.0,
                "gateway_ping_time": 15.0,
                "dns_resolution_time": 25.0,
                "memory_usage_percent": 35.0,
                "cpu_usage_percent": 20.0
            },
            "expected_result": {
                "is_anomaly": False,
                "anomaly_type": None
            }
        },
        
        "wifi_degradation": {
            "name": "WiFi信号衰减",
            "description": "WiFi信号质量下降，信号强度弱",
            "data": {
                "wlan0_wireless_quality": 45.0,
                "wlan0_signal_level": -70.0,
                "wlan0_noise_level": -75.0,
                "wlan0_rx_packets": 6000.0,
                "wlan0_tx_packets": 4500.0,
                "wlan0_rx_bytes": 1200000.0,
                "wlan0_tx_bytes": 1000000.0,
                "gateway_ping_time": 45.0,
                "dns_resolution_time": 60.0,
                "memory_usage_percent": 45.0,
                "cpu_usage_percent": 30.0
            },
            "expected_result": {
                "is_anomaly": True,
                "anomaly_type": "wifi_degradation"
            }
        },
        
        "network_latency": {
            "name": "网络延迟异常",
            "description": "网关ping和DNS解析时间过高",
            "data": {
                "wlan0_wireless_quality": 70.0,
                "wlan0_signal_level": -55.0,
                "wlan0_noise_level": -85.0,
                "wlan0_rx_packets": 12000.0,
                "wlan0_tx_packets": 9000.0,
                "wlan0_rx_bytes": 2200000.0,
                "wlan0_tx_bytes": 1800000.0,
                "gateway_ping_time": 150.0,
                "dns_resolution_time": 200.0,
                "memory_usage_percent": 40.0,
                "cpu_usage_percent": 25.0
            },
            "expected_result": {
                "is_anomaly": True,
                "anomaly_type": "network_latency"
            }
        },
        
        "system_stress": {
            "name": "系统压力异常",
            "description": "CPU和内存使用率过高",
            "data": {
                "wlan0_wireless_quality": 75.0,
                "wlan0_signal_level": -50.0,
                "wlan0_noise_level": -90.0,
                "wlan0_rx_packets": 14000.0,
                "wlan0_tx_packets": 11000.0,
                "wlan0_rx_bytes": 2800000.0,
                "wlan0_tx_bytes": 2300000.0,
                "gateway_ping_time": 30.0,
                "dns_resolution_time": 40.0,
                "memory_usage_percent": 95.0,
                "cpu_usage_percent": 90.0
            },
            "expected_result": {
                "is_anomaly": True,
                "anomaly_type": "system_stress"
            }
        }
    }
    
    return scenarios

def convert_to_float_array(data: Dict[str, float]) -> List[float]:
    """
    将网络数据转换为11维float数组（按照模型输入顺序）
    
    Args:
        data: 网络监控数据字典
        
    Returns:
        11维float数组
    """
    # 按照INPUT_FORMAT_SPECIFICATION.md中定义的顺序
    ordered_keys = [
        "wlan0_wireless_quality",     # index[0]
        "wlan0_signal_level",         # index[1] 
        "wlan0_noise_level",          # index[2]
        "wlan0_rx_packets",           # index[3]
        "wlan0_tx_packets",           # index[4]
        "wlan0_rx_bytes",             # index[5]
        "wlan0_tx_bytes",             # index[6]
        "gateway_ping_time",          # index[7]
        "dns_resolution_time",        # index[8]
        "memory_usage_percent",       # index[9]
        "cpu_usage_percent"           # index[10]
    ]
    
    float_array = []
    for key in ordered_keys:
        if key not in data:
            raise ValueError(f"Missing required field: {key}")
        float_array.append(float(data[key]))
    
    return float_array

def save_binary_data(float_array: List[float], filename: str) -> bool:
    """
    将float数组保存为二进制文件
    
    Args:
        float_array: 11维float数组
        filename: 输出文件名
        
    Returns:
        是否成功
    """
    try:
        with open(filename, 'wb') as f:
            # 每个float使用4字节（little-endian）
            for value in float_array:
                f.write(struct.pack('<f', value))
        return True
    except Exception as e:
        print(f"Error saving binary data: {e}")
        return False

def save_json_metadata(scenario_name: str, scenario_data: Dict[str, Any], 
                      float_array: List[float], filename: str) -> bool:
    """
    保存场景元数据到JSON文件
    
    Args:
        scenario_name: 场景名称
        scenario_data: 场景数据
        float_array: 转换后的float数组
        filename: 输出文件名
        
    Returns:
        是否成功
    """
    metadata = {
        "scenario_name": scenario_name,
        "description": scenario_data["description"],
        "original_data": scenario_data["data"],
        "float_array": float_array,
        "expected_result": scenario_data["expected_result"],
        "binary_format": {
            "data_type": "float32",
            "byte_order": "little-endian",
            "size_bytes": len(float_array) * 4,
            "element_count": len(float_array)
        }
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def validate_float_array(float_array: List[float]) -> bool:
    """验证float数组的合理性"""
    if len(float_array) != 11:
        print(f"Error: Expected 11 elements, got {len(float_array)}")
        return False
    
    # 检查数值范围（基本合理性检查）
    validation_rules = [
        (0, 100, "wlan0_wireless_quality"),      # [0, 100]
        (-100, -10, "wlan0_signal_level"),       # [-100, -10]
        (-100, -30, "wlan0_noise_level"),        # [-100, -30]
        (0, float('inf'), "wlan0_rx_packets"),   # [0, +∞]
        (0, float('inf'), "wlan0_tx_packets"),   # [0, +∞]
        (0, float('inf'), "wlan0_rx_bytes"),     # [0, +∞]
        (0, float('inf'), "wlan0_tx_bytes"),     # [0, +∞]
        (0, 5000, "gateway_ping_time"),          # [0, 5000]
        (0, 5000, "dns_resolution_time"),        # [0, 5000]
        (0, 100, "memory_usage_percent"),        # [0, 100]
        (0, 100, "cpu_usage_percent")            # [0, 100]
    ]
    
    for i, (min_val, max_val, field_name) in enumerate(validation_rules):
        value = float_array[i]
        if not (min_val <= value <= max_val):
            print(f"Warning: {field_name} value {value} out of expected range [{min_val}, {max_val}]")
    
    return True

def generate_test_data(scenario_name: Optional[str] = None, output_dir: str = ".") -> None:
    """
    生成测试数据
    
    Args:
        scenario_name: 指定场景名称，None表示生成所有场景
        output_dir: 输出目录
    """
    scenarios = create_test_scenarios()
    
    if scenario_name and scenario_name not in scenarios:
        print(f"Error: Unknown scenario '{scenario_name}'")
        print(f"Available scenarios: {list(scenarios.keys())}")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择要生成的场景
    if scenario_name:
        selected_scenarios = {scenario_name: scenarios[scenario_name]}
    else:
        selected_scenarios = scenarios
    
    print(f"=== 生成测试数据 ===")
    print(f"输出目录: {output_dir}")
    print(f"场景数量: {len(selected_scenarios)}")
    print()
    
    for name, scenario in selected_scenarios.items():
        print(f"🔄 生成场景: {name}")
        print(f"   描述: {scenario['description']}")
        
        # 转换为float数组
        try:
            float_array = convert_to_float_array(scenario["data"])
            
            # 验证数据
            if not validate_float_array(float_array):
                print(f"   ❌ 数据验证失败")
                continue
            
            # 生成文件名
            binary_filename = os.path.join(output_dir, f"{name}_input.bin")
            metadata_filename = os.path.join(output_dir, f"{name}_metadata.json")
            
            # 保存二进制数据
            if save_binary_data(float_array, binary_filename):
                print(f"   ✅ 二进制文件: {binary_filename}")
            else:
                print(f"   ❌ 二进制文件保存失败")
                continue
            
            # 保存元数据
            if save_json_metadata(name, scenario, float_array, metadata_filename):
                print(f"   ✅ 元数据文件: {metadata_filename}")
            else:
                print(f"   ❌ 元数据文件保存失败")
            
            # 显示文件大小
            file_size = os.path.getsize(binary_filename)
            print(f"   📊 文件大小: {file_size} 字节 ({file_size // 4} 个float)")
            
        except Exception as e:
            print(f"   ❌ 生成失败: {e}")
        
        print()

def main():
    """主函数"""
    print("🎯 **测试输入数据生成工具**")
    print("为移动设备DLC推理生成二进制格式的测试数据")
    print("=" * 60)
    
    if len(sys.argv) == 1:
        # 生成所有场景
        generate_test_data()
    elif len(sys.argv) == 2:
        # 生成指定场景
        scenario_name = sys.argv[1]
        generate_test_data(scenario_name)
    elif len(sys.argv) == 3:
        # 生成指定场景到指定目录
        scenario_name = sys.argv[1]
        output_dir = sys.argv[2]
        generate_test_data(scenario_name, output_dir)
    else:
        print("用法:")
        print(f"  {sys.argv[0]}                          # 生成所有测试场景")
        print(f"  {sys.argv[0]} <scenario_name>          # 生成指定场景")
        print(f"  {sys.argv[0]} <scenario_name> <dir>    # 生成到指定目录")
        print()
        print("可用场景:")
        scenarios = create_test_scenarios()
        for name, scenario in scenarios.items():
            print(f"  - {name}: {scenario['description']}")
        return
    
    print("💡 **使用生成的数据**:")
    print("   编译移动程序: chmod +x build_mobile_inference.sh && ./build_mobile_inference.sh")
    print("   运行推理: ./dlc_mobile_inference detector.dlc classifier.dlc normal_input.bin")
    print("   查看结果: cat inference_results.json")

if __name__ == "__main__":
    main() 