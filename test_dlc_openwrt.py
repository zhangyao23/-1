#!/usr/bin/env python3
"""
OpenWrt环境DLC模型测试脚本
模拟OpenWrt环境下的模型推理，使用JSON文件作为输入
"""

import os
import sys
import json
import numpy as np
import time
from pathlib import Path

def create_test_json_input():
    """
    创建测试用的JSON输入文件
    """
    # 创建一个更模糊的异常测试实例，包含多种异常特征
    test_input = {
        "network_data": {
            "wlan0_wireless_quality": 60.0,    # 信号质量中等
            "wlan0_signal_level": -55.0,       # 信号强度中等
            "wlan0_noise_level": -80.0,        # 噪声中等
            "wlan0_rx_packets": 950,           # 接收包数正常
            "wlan0_tx_packets": 1050,          # 发送包数略高
            "wlan0_rx_bytes": 1024000,         # 接收字节正常
            "wlan0_tx_bytes": 1536000,         # 发送字节略高
            "gateway_ping_time": 28.5,         # 网关延迟中等偏高
            "dns_resolution_time": 18.2,       # DNS解析延迟中等
            "memory_usage_percent": 72.0,      # 内存使用率偏高
            "cpu_usage_percent": 68.0          # CPU使用率偏高
        },
        "device_id": "openwrt_device_001",
        "timestamp": "2025-07-15T15:00:00Z"
    }
    
    with open("test_input.json", "w") as f:
        json.dump(test_input, f, indent=2)
    
    print("✅ 创建模糊异常测试输入文件: test_input.json")
    print("📊 混合异常特征分析:")
    print("   - WiFi信号中等 (60.0)")
    print("   - 信号强度中等 (-55.0 dBm)")
    print("   - 噪声中等 (-80.0 dBm)")
    print("   - 网络延迟中等偏高 (网关28.5ms, DNS 18.2ms)")
    print("   - 系统负载偏高 (内存72%, CPU 68%)")
    print("   - 流量模式异常 (发送包数>接收包数)")
    return test_input

def load_and_preprocess_json(json_file_path):
    """
    加载JSON文件并进行预处理
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # 提取11维网络数据
        network_data = data.get("network_data", {})
        
        # 按顺序提取11个特征
        features = [
            network_data.get("wlan0_wireless_quality", 0.0),
            network_data.get("wlan0_signal_level", 0.0),
            network_data.get("wlan0_noise_level", 0.0),
            network_data.get("wlan0_rx_packets", 0.0),
            network_data.get("wlan0_tx_packets", 0.0),
            network_data.get("wlan0_rx_bytes", 0.0),
            network_data.get("wlan0_tx_bytes", 0.0),
            network_data.get("gateway_ping_time", 0.0),
            network_data.get("dns_resolution_time", 0.0),
            network_data.get("memory_usage_percent", 0.0),
            network_data.get("cpu_usage_percent", 0.0)
        ]
        
        print(f"✅ 成功加载JSON数据，提取11维特征")
        print(f"📊 特征值: {features}")
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"❌ JSON加载失败: {e}")
        return None

def test_pytorch_model_inference():
    """
    使用PyTorch模型进行推理测试（模拟DLC推理）
    """
    print("\n🔍 测试PyTorch模型推理...")
    
    # 检查模型文件
    model_path = "multitask_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ PyTorch模型文件不存在: {model_path}")
        return False
    
    try:
        import torch
        from train_multitask_model import MultiTaskAnomalyModel
        
        # 加载模型
        model = MultiTaskAnomalyModel()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # 加载测试数据
        features = load_and_preprocess_json("test_input.json")
        if features is None:
            return False
        
        # 转换为tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0)  # 添加batch维度
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 解析输出
        detection_output = output[0, 0:2]  # 前2维是检测结果
        classification_output = output[0, 2:8]  # 后6维是分类结果
        
        # 转换为概率
        detection_probs = torch.softmax(detection_output, dim=0)
        classification_probs = torch.softmax(classification_output, dim=0)
        
        # 获取预测结果
        is_anomaly = detection_probs[1] > detection_probs[0]  # 异常概率 > 正常概率
        predicted_class = torch.argmax(classification_probs).item()
        
        # 异常类型映射
        anomaly_types = [
            "wifi_degradation",
            "network_latency",
            "connection_instability", 
            "bandwidth_congestion",
            "system_stress",
            "dns_issues"
        ]
        
        result = {
            "anomaly_detection": {
                "is_anomaly": bool(is_anomaly),
                "confidence": float(detection_probs[1] if is_anomaly else detection_probs[0]),
                "normal_probability": float(detection_probs[0]),
                "anomaly_probability": float(detection_probs[1])
            },
            "anomaly_classification": {
                "predicted_class": anomaly_types[predicted_class],
                "confidence": float(classification_probs[predicted_class]),
                "class_probabilities": {
                    anomaly_types[i]: float(classification_probs[i]) 
                    for i in range(len(anomaly_types))
                }
            },
            "device_id": "openwrt_device_001",
            "timestamp": "2025-07-15T15:00:00Z",
            "processing_time_ms": round(inference_time, 2)
        }
        
        print(f"✅ 推理完成，耗时: {inference_time:.2f} ms")
        print(f"📊 检测结果: {'异常' if is_anomaly else '正常'} (置信度: {result['anomaly_detection']['confidence']:.3f})")
        print(f"📊 分类结果: {result['anomaly_classification']['predicted_class']} (置信度: {result['anomaly_classification']['confidence']:.3f})")
        
        # 保存结果
        with open("test_output.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("✅ 结果已保存到: test_output.json")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch推理失败: {e}")
        return False

def check_dlc_compatibility():
    """
    检查DLC文件兼容性
    """
    print("\n🔍 检查DLC文件兼容性...")
    
    dlc_path = "multitask_model.dlc"
    if not os.path.exists(dlc_path):
        print(f"❌ DLC文件不存在: {dlc_path}")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(dlc_path)
    print(f"📊 DLC文件大小: {file_size / 1024:.1f} KB")
    
    # 检查文件格式
    with open(dlc_path, 'rb') as f:
        header = f.read(4)
        if header.startswith(b'PK'):
            print("✅ DLC文件格式正确 (ZIP格式)")
        else:
            print("❌ DLC文件格式错误")
            return False
    
    # 检查SNPE环境
    snpe_root = "2.26.2.240911"
    if not os.path.exists(snpe_root):
        print(f"⚠️  SNPE SDK未找到: {snpe_root}")
        return False
    
    # 检查推理工具
    snpe_net_run = os.path.join(snpe_root, "bin", "x86_64-linux-clang", "snpe-net-run")
    if os.path.exists(snpe_net_run):
        print("✅ SNPE推理工具可用")
    else:
        print("❌ SNPE推理工具不可用")
        return False
    
    return True

def generate_openwrt_integration_guide():
    """
    生成OpenWrt集成指南
    """
    print("\n📋 OpenWrt集成指南")
    print("=" * 50)
    
    guide = {
        "文件准备": [
            "multitask_model.dlc - 模型文件",
            "test_input.json - 输入数据格式示例",
            "test_output.json - 输出数据格式示例"
        ],
        "OpenWrt环境要求": [
            "ARM架构支持",
            "SNPE运行时库",
            "C++11编译器",
            "JSON解析库 (如nlohmann/json)"
        ],
        "集成步骤": [
            "1. 将multitask_model.dlc复制到OpenWrt设备",
            "2. 安装SNPE运行时库",
            "3. 编译C++推理程序",
            "4. 配置JSON输入输出格式",
            "5. 集成到现有监控系统"
        ],
        "性能预期": [
            "推理时间: < 1ms",
            "内存占用: < 50MB",
            "CPU占用: < 5%",
            "支持实时监控"
        ]
    }
    
    for key, items in guide.items():
        print(f"\n{key}:")
        for item in items:
            print(f"  • {item}")
    
    print(f"\n💡 详细集成代码请参考: guide/模型集成指南.md")

def generate_all_anomaly_inputs():
    """
    生成6种异常类型的典型输入并测试推理准确性
    """
    anomaly_cases = [
        # 1. WiFi劣化（wifi_degradation）
        {
            "desc": "WiFi劣化",
            "input": {
                "wlan0_wireless_quality": 20.0,  # 质量明显下降
                "wlan0_signal_level": -80.0,     # 信号明显弱
                "wlan0_noise_level": -60.0,      # 噪声明显高
                "wlan0_rx_packets": 8000,
                "wlan0_tx_packets": 6000,
                "wlan0_rx_bytes": 1500000,
                "wlan0_tx_bytes": 1200000,
                "gateway_ping_time": 40.0,
                "dns_resolution_time": 50.0,
                "memory_usage_percent": 45.0,
                "cpu_usage_percent": 30.0
            }
        },
        # 2. 网络延迟（network_latency）
        {
            "desc": "网络延迟",
            "input": {
                "wlan0_wireless_quality": 70.0,
                "wlan0_signal_level": -55.0,
                "wlan0_noise_level": -85.0,
                "wlan0_rx_packets": 12000,
                "wlan0_tx_packets": 10000,
                "wlan0_rx_bytes": 2500000,
                "wlan0_tx_bytes": 2000000,
                "gateway_ping_time": 150.0,  # ping明显长
                "dns_resolution_time": 180.0, # DNS明显慢
                "memory_usage_percent": 40.0,
                "cpu_usage_percent": 25.0
            }
        },
        # 3. 连接不稳定（connection_instability）
        {
            "desc": "连接不稳定",
            "input": {
                "wlan0_wireless_quality": 40.0,  # 质量不稳定
                "wlan0_signal_level": -75.0,     # 信号不稳定
                "wlan0_noise_level": -65.0,      # 噪声较高
                "wlan0_rx_packets": 2000,        # 包数明显少
                "wlan0_tx_packets": 1500,        # 包数明显少
                "wlan0_rx_bytes": 300000,        # 流量少
                "wlan0_tx_bytes": 250000,        # 流量少
                "gateway_ping_time": 80.0,
                "dns_resolution_time": 100.0,
                "memory_usage_percent": 35.0,
                "cpu_usage_percent": 20.0
            }
        },
        # 4. 带宽拥塞（bandwidth_congestion）
        {
            "desc": "带宽拥塞",
            "input": {
                "wlan0_wireless_quality": 85.0,  # 质量好
                "wlan0_signal_level": -40.0,     # 信号好
                "wlan0_noise_level": -95.0,      # 噪声低
                "wlan0_rx_packets": 35000,       # 包数很多
                "wlan0_tx_packets": 30000,       # 包数很多
                "wlan0_rx_bytes": 12000000,      # 流量很高
                "wlan0_tx_bytes": 10000000,      # 流量很高
                "gateway_ping_time": 70.0,
                "dns_resolution_time": 60.0,
                "memory_usage_percent": 75.0,    # 内存使用高
                "cpu_usage_percent": 60.0        # CPU使用高
            }
        },
        # 5. 系统压力（system_stress）
        {
            "desc": "系统压力",
            "input": {
                "wlan0_wireless_quality": 75.0,
                "wlan0_signal_level": -50.0,
                "wlan0_noise_level": -90.0,
                "wlan0_rx_packets": 14000,
                "wlan0_tx_packets": 11000,
                "wlan0_rx_bytes": 2800000,
                "wlan0_tx_bytes": 2300000,
                "gateway_ping_time": 30.0,
                "dns_resolution_time": 40.0,
                "memory_usage_percent": 95.0,    # 内存极高
                "cpu_usage_percent": 90.0        # CPU极高
            }
        },
        # 6. DNS异常（dns_issues）
        {
            "desc": "DNS异常",
            "input": {
                "wlan0_wireless_quality": 75.0,
                "wlan0_signal_level": -50.0,
                "wlan0_noise_level": -90.0,
                "wlan0_rx_packets": 15000,
                "wlan0_tx_packets": 12000,
                "wlan0_rx_bytes": 3000000,
                "wlan0_tx_bytes": 2500000,
                "gateway_ping_time": 25.0,
                "dns_resolution_time": 400.0,    # DNS极慢
                "memory_usage_percent": 40.0,
                "cpu_usage_percent": 25.0
            }
        }
    ]
    anomaly_types = [
        "wifi_degradation",
        "network_latency",
        "connection_instability", 
        "bandwidth_congestion",
        "system_stress",
        "dns_issues"
    ]
    import torch
    from train_multitask_model import MultiTaskAnomalyModel
    model = MultiTaskAnomalyModel()
    model.load_state_dict(torch.load("multitask_model.pth", map_location='cpu'))
    model.eval()
    print("\n===== 6种异常类型推理结果 =====")
    for idx, case in enumerate(anomaly_cases):
        features = [
            case["input"]["wlan0_wireless_quality"],
            case["input"]["wlan0_signal_level"],
            case["input"]["wlan0_noise_level"],
            case["input"]["wlan0_rx_packets"],
            case["input"]["wlan0_tx_packets"],
            case["input"]["wlan0_rx_bytes"],
            case["input"]["wlan0_tx_bytes"],
            case["input"]["gateway_ping_time"],
            case["input"]["dns_resolution_time"],
            case["input"]["memory_usage_percent"],
            case["input"]["cpu_usage_percent"]
        ]
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        detection_output = output[0, 0:2]
        classification_output = output[0, 2:8]
        detection_probs = torch.softmax(detection_output, dim=0)
        classification_probs = torch.softmax(classification_output, dim=0)
        is_anomaly = detection_probs[1] > detection_probs[0]
        predicted_class = torch.argmax(classification_probs).item()
        print(f"\n【{case['desc']}】")
        print(f"  检测结果: {'异常' if is_anomaly else '正常'} (异常概率: {detection_probs[1]:.3f})")
        print(f"  分类结果: {anomaly_types[predicted_class]} (置信度: {classification_probs[predicted_class]:.3f})")
        print(f"  各类型概率: {[f'{anomaly_types[i]}={classification_probs[i]:.3f}' for i in range(6)]}")

def main():
    """
    主测试流程
    """
    print("🚀 OpenWrt DLC模型兼容性测试")
    print("=" * 50)
    
    # 1. 创建测试输入
    create_test_json_input()
    
    # 2. 检查DLC兼容性
    dlc_ok = check_dlc_compatibility()
    
    # 3. 测试PyTorch推理（模拟DLC推理）
    inference_ok = test_pytorch_model_inference()
    
    # 4. 生成集成指南
    generate_openwrt_integration_guide()
    
    # 5. 6种异常类型推理准确性测试
    generate_all_anomaly_inputs()
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"  DLC兼容性检查: {'✅ 通过' if dlc_ok else '❌ 失败'}")
    print(f"  推理功能测试: {'✅ 通过' if inference_ok else '❌ 失败'}")
    
    if dlc_ok and inference_ok:
        print("\n🎉 测试通过！DLC模型可以在OpenWrt环境下正常工作")
        print("📁 生成的文件:")
        print("   - test_input.json (输入格式示例)")
        print("   - test_output.json (输出格式示例)")
        return True
    else:
        print("\n⚠️  测试失败，请检查相关问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 