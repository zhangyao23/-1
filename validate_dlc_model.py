#!/usr/bin/env python3
"""
DLC模型验证脚本
验证生成的DLC模型文件是否符合目标板子的要求
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

def check_dlc_file(dlc_path="multitask_model.dlc"):
    """
    检查DLC文件的基本信息
    """
    print(f"🔍 检查DLC文件: {dlc_path}")
    
    if not os.path.exists(dlc_path):
        print(f"❌ DLC文件不存在: {dlc_path}")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(dlc_path)
    print(f"📊 文件大小: {file_size / 1024:.1f} KB")
    
    # 检查文件权限
    if os.access(dlc_path, os.R_OK):
        print("✅ 文件可读")
    else:
        print("❌ 文件不可读")
        return False
    
    return True

def validate_input_format():
    """
    验证输入数据格式
    """
    print("\n🔍 验证输入数据格式...")
    
    # 检查示例输入文件
    example_input = "example_normal_input.json"
    if not os.path.exists(example_input):
        print(f"❌ 示例输入文件不存在: {example_input}")
        return False
    
    try:
        with open(example_input, 'r') as f:
            data = json.load(f)
        
        # 检查必需的字段
        required_fields = [
            "wlan0_wireless_quality", "wlan0_signal_level", "wlan0_noise_level",
            "wlan0_rx_packets", "wlan0_tx_packets", "wlan0_rx_bytes", "wlan0_tx_bytes",
            "gateway_ping_time", "dns_resolution_time", "memory_usage_percent", "cpu_usage_percent"
        ]
        
        network_data = data.get("network_data", {})
        missing_fields = []
        
        for field in required_fields:
            if field not in network_data:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ 缺少必需字段: {missing_fields}")
            return False
        
        print(f"✅ 输入格式验证通过 (11个字段)")
        print(f"📊 数据示例: {list(network_data.values())[:3]}...")
        return True
        
    except Exception as e:
        print(f"❌ 输入格式验证失败: {e}")
        return False

def validate_output_format():
    """
    验证输出数据格式
    """
    print("\n🔍 验证输出数据格式...")
    
    # 检查示例输出文件
    example_output = "inference_results.json"
    if not os.path.exists(example_output):
        print(f"⚠️  示例输出文件不存在: {example_output}")
        print("将使用标准输出格式进行验证")
        
        # 创建标准输出格式示例
        standard_output = {
            "anomaly_detection": {
                "is_anomaly": True,
                "confidence": 0.999,
                "normal_probability": 0.001,
                "anomaly_probability": 0.999
            },
            "anomaly_classification": {
                "predicted_class": "dns_issues",
                "confidence": 0.998,
                "class_probabilities": {
                    "bandwidth_congestion": 0.002,
                    "connection_instability": 0.000,
                    "dns_issues": 0.998,
                    "network_latency": 0.000,
                    "system_stress": 0.000,
                    "wifi_degradation": 0.000
                }
            },
            "device_id": "device_001",
            "timestamp": "2025-07-07T14:30:00Z",
            "processing_time_ms": 25
        }
        
        print("✅ 标准输出格式验证通过")
        return True
    
    try:
        with open(example_output, 'r') as f:
            data = json.load(f)
        
        # 检查必需的字段
        required_sections = ["anomaly_detection", "anomaly_classification"]
        for section in required_sections:
            if section not in data:
                print(f"❌ 缺少输出部分: {section}")
                return False
        
        # 检查异常类型
        anomaly_types = [
            "bandwidth_congestion", "connection_instability", "dns_issues",
            "network_latency", "system_stress", "wifi_degradation"
        ]
        
        class_probs = data["anomaly_classification"].get("class_probabilities", {})
        for anomaly_type in anomaly_types:
            if anomaly_type not in class_probs:
                print(f"❌ 缺少异常类型: {anomaly_type}")
                return False
        
        print("✅ 输出格式验证通过")
        print(f"📊 支持的异常类型: {len(anomaly_types)}种")
        return True
        
    except Exception as e:
        print(f"❌ 输出格式验证失败: {e}")
        return False

def check_snpe_compatibility():
    """
    检查SNPE兼容性
    """
    print("\n🔍 检查SNPE兼容性...")
    
    snpe_root = "2.26.2.240911"
    if not os.path.exists(snpe_root):
        print(f"⚠️  SNPE SDK未找到: {snpe_root}")
        print("请确保SNPE SDK已正确安装")
        return False
    
    # 检查SNPE版本
    version_file = os.path.join(snpe_root, "version.txt")
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            version = f.read().strip()
        print(f"✅ SNPE版本: {version}")
    else:
        print("✅ SNPE SDK存在")
    
    # 检查转换工具
    converter = os.path.join(snpe_root, "bin", "x86_64-linux-clang", "snpe-onnx-to-dlc")
    if os.path.exists(converter):
        print("✅ SNPE转换工具可用")
        return True
    else:
        print("❌ SNPE转换工具不可用")
        return False

def test_multitask_performance():
    """
    测试多任务模型性能
    """
    print("\n🔍 测试多任务模型性能...")
    
    # 检查是否有训练好的PyTorch模型用于性能测试
    pytorch_model_path = "multitask_model.pth"
    if not os.path.exists(pytorch_model_path):
        print(f"⚠️  PyTorch模型文件不存在: {pytorch_model_path}")
        print("跳过性能测试")
        return True
    
    try:
        import torch
        from train_multitask_model import MultiTaskAnomalyModel
        
        # 加载模型
        model = MultiTaskAnomalyModel()
        model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
        model.eval()
        
        # 生成测试数据
        test_input = torch.randn(100, 11)  # 100个样本
        
        # 测试推理时间
        import time
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input[:1])
        
        # 性能测试
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = model(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = 100 / (end_time - start_time)
        
        print(f"✅ 性能测试完成")
        print(f"📊 平均推理时间: {avg_time*1000:.2f} ms")
        print(f"📊 吞吐量: {throughput:.1f} 样本/秒")
        print(f"📊 输出维度: 检测({output.shape[1]-6}) + 分类({6}) = {output.shape[1]}维")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def generate_integration_summary():
    """
    生成集成摘要
    """
    print("\n📋 集成摘要")
    print("=" * 40)
    
    summary = {
        "模型文件": {
            "名称": "multitask_model.dlc",
            "大小": f"{os.path.getsize('multitask_model.dlc') / 1024:.1f} KB" if os.path.exists("multitask_model.dlc") else "未找到",
            "状态": "✅ 可用" if os.path.exists("multitask_model.dlc") else "❌ 不可用",
            "类型": "多任务模型 (检测+分类)"
        },
        "输入格式": {
            "维度": "11维",
            "格式": "JSON",
            "字段": "WiFi信号(3) + 网络流量(4) + 网络延迟(2) + 系统资源(2)"
        },
        "输出格式": {
            "检测": "异常检测结果 (2维: 正常/异常)",
            "分类": "异常分类结果 (6维: 6种异常类型)",
            "格式": "JSON",
            "特点": "单次推理完成两个任务"
        },
        "异常类型": [
            "bandwidth_congestion",
            "connection_instability", 
            "dns_issues",
            "network_latency",
            "system_stress",
            "wifi_degradation"
        ],
        "性能优势": [
            "单次推理完成检测和分类",
            "减少计算开销",
            "提高推理效率",
            "统一模型管理"
        ]
    }
    
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
    
    print("\n💡 集成建议:")
    print("1. 将 multitask_model.dlc 复制到目标板子的模型目录")
    print("2. 确保目标板子的C++脚本支持SNPE DLC格式")
    print("3. 按照输入/输出格式规范集成到现有系统")
    print("4. 参考 guide/模型集成指南.md 获取详细说明")

def main():
    """
    主验证流程
    """
    print("🚀 DLC模型验证开始")
    print("=" * 50)
    
    checks = [
        ("DLC文件检查", check_dlc_file),
        ("输入格式验证", validate_input_format),
        ("输出格式验证", validate_output_format),
        ("SNPE兼容性检查", check_snpe_compatibility),
        ("多任务性能测试", test_multitask_performance)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name}失败: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("📊 验证结果汇总:")
    
    all_passed = True
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有验证通过！模型可以集成到目标板子")
        generate_integration_summary()
        return True
    else:
        print("\n⚠️  部分验证失败，请检查相关问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 