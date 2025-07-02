#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试场景脚本

提供多种预设的测试场景，方便快速验证AI检测系统的性能
包括正常场景和各种异常场景的测试数据
"""

import os
import sys
import json
from pathlib import Path

# 将src目录添加到Python路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_processor.feature_extractor import FeatureExtractor
from anomaly_detector.anomaly_engine import AnomalyDetectionEngine
from logger.system_logger import SystemLogger
from ai_models.autoencoder_model import AutoencoderModel
from ai_models.error_classifier import ErrorClassifier

# 预设测试场景
TEST_SCENARIOS = {
    "normal_1": {
        "name": "正常场景1 - 良好网络状态",
        "description": "所有指标都在正常范围内",
        "data": {
            "wlan0_wireless_quality": 75.0,
            "wlan0_wireless_level": -45.0,
            "wlan0_packet_loss_rate": 0.005,
            "wlan0_send_rate_bps": 1000000.0,
            "wlan0_recv_rate_bps": 2000000.0,
            "tcp_retrans_segments": 2,
            "gateway_ping_time": 8.5,
            "dns_response_time": 15.0,
            "tcp_connection_count": 25,
            "cpu_percent": 12.0,
            "memory_percent": 35.0
        }
    },
    
    "normal_2": {
        "name": "正常场景2 - 中等网络状态",
        "description": "指标稍高但仍在正常范围",
        "data": {
            "wlan0_wireless_quality": 65.0,
            "wlan0_wireless_level": -60.0,
            "wlan0_packet_loss_rate": 0.02,
            "wlan0_send_rate_bps": 500000.0,
            "wlan0_recv_rate_bps": 1200000.0,
            "tcp_retrans_segments": 5,
            "gateway_ping_time": 15.0,
            "dns_response_time": 30.0,
            "tcp_connection_count": 35,
            "cpu_percent": 20.0,
            "memory_percent": 50.0
        }
    },
    
    "signal_degradation": {
        "name": "信号衰减异常",
        "description": "WiFi信号质量严重下降",
        "data": {
            "wlan0_wireless_quality": 25.0,  # 信号质量很差
            "wlan0_wireless_level": -85.0,   # 信号强度很弱
            "wlan0_packet_loss_rate": 0.15,  # 丢包率高
            "wlan0_send_rate_bps": 100000.0,
            "wlan0_recv_rate_bps": 200000.0,
            "tcp_retrans_segments": 15,
            "gateway_ping_time": 45.0,
            "dns_response_time": 80.0,
            "tcp_connection_count": 20,
            "cpu_percent": 15.0,
            "memory_percent": 40.0
        }
    },
    
    "network_congestion": {
        "name": "网络拥塞异常",
        "description": "网络延迟和丢包严重",
        "data": {
            "wlan0_wireless_quality": 60.0,
            "wlan0_wireless_level": -55.0,
            "wlan0_packet_loss_rate": 0.08,   # 高丢包率
            "wlan0_send_rate_bps": 200000.0,  # 低传输速率
            "wlan0_recv_rate_bps": 300000.0,
            "tcp_retrans_segments": 25,       # 大量重传
            "gateway_ping_time": 100.0,       # 高延迟
            "dns_response_time": 150.0,
            "tcp_connection_count": 50,
            "cpu_percent": 18.0,
            "memory_percent": 45.0
        }
    },
    
    "resource_overload": {
        "name": "资源过载异常",
        "description": "CPU和内存使用率过高",
        "data": {
            "wlan0_wireless_quality": 70.0,
            "wlan0_wireless_level": -50.0,
            "wlan0_packet_loss_rate": 0.03,
            "wlan0_send_rate_bps": 800000.0,
            "wlan0_recv_rate_bps": 1000000.0,
            "tcp_retrans_segments": 8,
            "gateway_ping_time": 20.0,
            "dns_response_time": 35.0,
            "tcp_connection_count": 40,
            "cpu_percent": 85.0,              # 高CPU使用率
            "memory_percent": 90.0            # 高内存使用率
        }
    },
    
    "connection_timeout": {
        "name": "连接超时异常",
        "description": "网络响应时间过长",
        "data": {
            "wlan0_wireless_quality": 55.0,
            "wlan0_wireless_level": -70.0,
            "wlan0_packet_loss_rate": 0.05,
            "wlan0_send_rate_bps": 300000.0,
            "wlan0_recv_rate_bps": 400000.0,
            "tcp_retrans_segments": 12,
            "gateway_ping_time": 200.0,       # 极高延迟
            "dns_response_time": 300.0,       # DNS超时
            "tcp_connection_count": 15,
            "cpu_percent": 25.0,
            "memory_percent": 55.0
        }
    }
}

def load_config():
    """加载系统配置"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_test_scenario(scenario_key: str, engine, extractor):
    """运行指定的测试场景"""
    if scenario_key not in TEST_SCENARIOS:
        print(f"❌ 未找到测试场景: {scenario_key}")
        return
    
    scenario = TEST_SCENARIOS[scenario_key]
    print(f"\n🧪 测试场景: {scenario['name']}")
    print(f"📝 描述: {scenario['description']}")
    print("-" * 60)
    
    # 显示输入数据
    print("📊 输入数据:")
    for key, value in scenario['data'].items():
        print(f"  {key}: {value}")
    
    print("\n🔄 正在进行AI检测...")
    
    # 进行检测
    try:
        # 提取特征
        feature_vector = extractor.extract_features(scenario['data'])
        feature_names = extractor.get_feature_names()
        
        if feature_vector.size == 0:
            print("❌ 特征提取失败")
            return
            
        # 使用检测引擎
        is_anomaly, details = engine.detect_anomaly_from_vector(feature_vector, feature_names)
        
        print("\n" + "=" * 50)
        print("🎯 检测结果")
        print("=" * 50)
        
        if is_anomaly:
            print(f"状态: ⚠️  检测到异常!")
            predicted_class = details.get('predicted_class', 'N/A')
            confidence = details.get('confidence', 0.0)
            print(f"预测类型: {predicted_class}")
            print(f"置信度: {confidence:.1%}")
            
            error = details.get('reconstruction_error', 'N/A')
            threshold = details.get('threshold', 'N/A')
            if error != 'N/A':
                print(f"重构误差: {error:.6f}")
            if threshold != 'N/A':
                print(f"异常阈值: {threshold:.6f}")
        else:
            print(f"状态: ✅ 正常")
            error = details.get('reconstruction_error', 'N/A')
            threshold = details.get('threshold', 'N/A')
            if error != 'N/A':
                print(f"重构误差: {error:.6f}")
            if threshold != 'N/A':
                print(f"异常阈值: {threshold:.6f}")
            
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 检测过程中发生错误: {e}")

def get_default_inputs():
    """从simulation_inputs.json加载"正常"情况作为默认值"""
    inputs_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation_inputs.json')
    try:
        with open(inputs_path, 'r', encoding='utf-8') as f:
            for case in json.load(f):
                if "正常" in case.get("name", ""):
                    return case.get("data", {})
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或损坏，则使用硬编码的后备值
        return {
            'wlan0_wireless_quality': 70.0, 'wlan0_wireless_level': -55.0,
            'wlan0_packet_loss_rate': 0.01, 'wlan0_send_rate_bps': 500000.0,
            'wlan0_recv_rate_bps': 1500000.0, 'tcp_retrans_segments': 5,
            'gateway_ping_time': 12.5, 'dns_response_time': 25.0,
            'tcp_connection_count': 30, 'cpu_percent': 15.0, 'memory_percent': 45.0
        }
    return {}

def main():
    """主函数"""
    print("🚀 AI异常检测系统 - 测试场景脚本")
    print("=" * 60)
    
    # 初始化所有组件
    print("⏳ 正在初始化AI检测器...")
    try:
        config = load_config()
        logger = SystemLogger(config['logging'])
        
        # 将日志级别设为WARNING，以获得更干净的输出
        logger.set_log_level('WARNING')
        
        extractor = FeatureExtractor(config['data_collection']['metrics'], logger)
        autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
        classifier = ErrorClassifier(config['ai_models']['classifier'], logger)
        
        engine = AnomalyDetectionEngine(
            config=config['anomaly_detection'],
            autoencoder=autoencoder, error_classifier=classifier,
            buffer_manager=None, logger=logger
        )
        
        # 使用正常样本校准特征提取器
        print("⚡ 正在校准特征提取器...")
        normal_baseline_data = get_default_inputs()
        if normal_baseline_data:
            extractor.extract_features(normal_baseline_data)
        
        print("✅ AI检测器初始化成功")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 显示可用场景
    print("\n📋 可用的测试场景:")
    for i, (key, scenario) in enumerate(TEST_SCENARIOS.items(), 1):
        print(f"  {i}. {key} - {scenario['name']}")
    
    print("\n📖 使用方法:")
    print("  python3 scripts/test_scenarios.py                    # 运行所有场景")
    print("  python3 scripts/test_scenarios.py normal_1           # 运行指定场景")
    print("  python3 scripts/test_scenarios.py normal_1 signal_degradation  # 运行多个场景")
    
    # 处理命令行参数
    scenarios_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(TEST_SCENARIOS.keys())
    
    # 运行测试场景
    for scenario_key in scenarios_to_run:
        run_test_scenario(scenario_key, engine, extractor)
    
    print(f"\n🎉 测试完成！共运行了 {len(scenarios_to_run)} 个场景。")

if __name__ == "__main__":
    main() 