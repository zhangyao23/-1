#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析所有测试场景的重构误差分布
用于确定最佳异常检测阈值
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_processor.feature_extractor import FeatureExtractor
from src.ai_models.autoencoder_model import AutoencoderModel

def load_test_scenarios():
    """加载所有测试场景"""
    scenarios_file = os.path.join(project_root, 'data', 'test_scenarios.json')
    
    if not os.path.exists(scenarios_file):
        # 如果文件不存在，创建基本的测试场景
        scenarios = [
            {"id": 1, "name": "正常网络状态", "type": "normal", "description": "一切正常的网络环境", "expected": "normal"},
            {"id": 2, "name": "轻负载状态", "type": "normal_light_load", "description": "网络使用量较低的正常状态", "expected": "normal"},
            {"id": 3, "name": "高负载正常状态", "type": "normal_high_load", "description": "网络使用量高但仍在正常范围", "expected": "normal"},
            {"id": 4, "name": "WiFi信号衰减", "type": "signal_degradation", "description": "用户离路由器太远，信号质量下降", "expected": "anomaly"},
            {"id": 5, "name": "带宽饱和", "type": "bandwidth_saturation", "description": "网络带宽被大量使用占满", "expected": "anomaly"},
            {"id": 6, "name": "DDoS攻击", "type": "ddos_attack", "description": "遭受分布式拒绝服务攻击", "expected": "anomaly"},
            {"id": 7, "name": "DNS配置错误", "type": "dns_misconfiguration", "description": "DNS服务器配置不当", "expected": "anomaly"},
            {"id": 8, "name": "CPU过载", "type": "cpu_overload", "description": "CPU使用率过高影响网络性能", "expected": "anomaly"}
        ]
    else:
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
    
    return scenarios

def create_scenario_data(scenario_type: str) -> Dict:
    """根据场景类型创建测试数据"""
    base_data = {
        'wlan0_wireless_quality': 85.0,
        'wlan0_signal_level': -45.0,
        'wlan0_noise_level': -90.0,
        'wlan0_rx_packets': 50000,
        'wlan0_tx_packets': 30000,
        'wlan0_rx_bytes': 75000000,
        'wlan0_tx_bytes': 25000000,
        'gateway_ping_time': 8.5,
        'dns_resolution_time': 15.2,
        'memory_usage_percent': 35.0,
        'cpu_usage_percent': 12.0
    }
    
    # 根据场景类型调整数据
    if scenario_type in ['normal', 'normal_light_load']:
        return base_data
    elif scenario_type == 'normal_high_load':
        base_data.update({
            'wlan0_rx_packets': 120000,
            'wlan0_tx_packets': 80000,
            'cpu_usage_percent': 45.0,
            'memory_usage_percent': 60.0
        })
    elif scenario_type == 'signal_degradation':
        base_data.update({
            'wlan0_wireless_quality': 35.0,
            'wlan0_signal_level': -75.0,
            'wlan0_noise_level': -85.0
        })
    elif scenario_type == 'bandwidth_saturation':
        base_data.update({
            'wlan0_rx_bytes': 150000000,
            'wlan0_tx_bytes': 80000000,
            'gateway_ping_time': 25.0
        })
    elif scenario_type == 'ddos_attack':
        base_data.update({
            'wlan0_rx_packets': 500000,
            'wlan0_tx_packets': 200000,
            'gateway_ping_time': 100.0,
            'cpu_usage_percent': 85.0
        })
    elif scenario_type == 'dns_misconfiguration':
        base_data.update({
            'dns_resolution_time': 5000.0,
            'gateway_ping_time': 12.0
        })
    elif scenario_type == 'cpu_overload':
        base_data.update({
            'cpu_usage_percent': 95.0,
            'memory_usage_percent': 80.0,
            'gateway_ping_time': 30.0
        })
    
    return base_data

def analyze_reconstruction_errors():
    """分析所有场景的重构误差"""
    try:
        # 加载配置
        config_path = os.path.join(project_root, 'config', 'system_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建简单的日志对象
        class SimpleLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass  # 关闭debug输出
            def warning(self, msg): print(f"WARNING: {msg}")
        
        logger = SimpleLogger()
        
        # 初始化组件
        real_metrics = [
            'wlan0_wireless_quality', 'wlan0_signal_level', 'wlan0_noise_level',
            'wlan0_rx_packets', 'wlan0_tx_packets', 'wlan0_rx_bytes', 'wlan0_tx_bytes',
            'gateway_ping_time', 'dns_resolution_time', 'memory_usage_percent', 'cpu_usage_percent'
        ]
        
        scaler_path = os.path.join(config['ai_models']['autoencoder']['model_path'], 'autoencoder_scaler.pkl')
        feature_extractor = FeatureExtractor(real_metrics, logger, scaler_path=scaler_path)
        autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
        
        # 加载测试场景
        scenarios = load_test_scenarios()
        
        # 分析结果
        normal_errors = []
        anomaly_errors = []
        all_results = []
        
        print("🔍 分析所有场景的重构误差...")
        print("=" * 60)
        
        for scenario in scenarios:
            scenario_data = create_scenario_data(scenario['type'])
            features = feature_extractor.extract_features(scenario_data)
            result = autoencoder.predict(features)
            
            error = result['reconstruction_error']
            expected = scenario['expected']
            
            all_results.append({
                'name': scenario['name'],
                'type': scenario['type'],
                'expected': expected,
                'error': error
            })
            
            if expected == 'normal':
                normal_errors.append(error)
            else:
                anomaly_errors.append(error)
            
            status = "🟢 正常" if expected == 'normal' else "🔴 异常"
            print(f"{status} {scenario['name']:<25} 误差: {error:>10.2f}")
        
        # 统计分析
        print("\n📊 重构误差分布分析:")
        print("=" * 60)
        
        print(f"正常场景 ({len(normal_errors)}个):")
        if normal_errors:
            print(f"  最小误差: {min(normal_errors):.2f}")
            print(f"  最大误差: {max(normal_errors):.2f}")
            print(f"  平均误差: {np.mean(normal_errors):.2f}")
            print(f"  标准差:   {np.std(normal_errors):.2f}")
        
        print(f"\n异常场景 ({len(anomaly_errors)}个):")
        if anomaly_errors:
            print(f"  最小误差: {min(anomaly_errors):.2f}")
            print(f"  最大误差: {max(anomaly_errors):.2f}")
            print(f"  平均误差: {np.mean(anomaly_errors):.2f}")
            print(f"  标准差:   {np.std(anomaly_errors):.2f}")
        
        # 推荐阈值
        print(f"\n🎯 阈值推荐:")
        print("=" * 60)
        
        if normal_errors and anomaly_errors:
            max_normal = max(normal_errors)
            min_anomaly = min(anomaly_errors)
            
            print(f"正常场景最大误差: {max_normal:.2f}")
            print(f"异常场景最小误差: {min_anomaly:.2f}")
            
            if max_normal < min_anomaly:
                # 有明确分界线
                recommended_threshold = (max_normal + min_anomaly) / 2
                print(f"✅ 推荐阈值: {recommended_threshold:.2f} (完美分类)")
            else:
                # 有重叠，需要找平衡点
                recommended_threshold = np.percentile(normal_errors, 95)
                print(f"⚠️  存在误差重叠，推荐阈值: {recommended_threshold:.2f}")
                print(f"   (基于正常场景95%分位数)")
        
        return all_results, normal_errors, anomaly_errors
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    analyze_reconstruction_errors() 