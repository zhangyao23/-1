#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
详细分析各场景的特征值分布
找出异常场景设计的问题所在
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_processor.feature_extractor import FeatureExtractor

def load_test_scenarios():
    """加载所有测试场景"""
    scenarios = [
        {"id": 1, "name": "正常网络状态", "type": "normal", "expected": "normal"},
        {"id": 2, "name": "轻负载状态", "type": "normal_light_load", "expected": "normal"},
        {"id": 3, "name": "高负载正常状态", "type": "normal_high_load", "expected": "normal"},
        {"id": 4, "name": "WiFi信号衰减", "type": "signal_degradation", "expected": "anomaly"},
        {"id": 5, "name": "带宽饱和", "type": "bandwidth_saturation", "expected": "anomaly"},
        {"id": 6, "name": "DDoS攻击", "type": "ddos_attack", "expected": "anomaly"},
        {"id": 7, "name": "DNS配置错误", "type": "dns_misconfiguration", "expected": "anomaly"},
        {"id": 8, "name": "CPU过载", "type": "cpu_overload", "expected": "anomaly"}
    ]
    return scenarios

def create_scenario_data(scenario_type: str):
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

def analyze_feature_distributions():
    """分析各场景的特征分布"""
    try:
        # 加载配置
        config_path = os.path.join(project_root, 'config', 'system_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建简单的日志对象
        class SimpleLogger:
            def info(self, msg): pass
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass
            def warning(self, msg): pass
        
        logger = SimpleLogger()
        
        # 初始化特征提取器
        real_metrics = [
            'wlan0_wireless_quality', 'wlan0_signal_level', 'wlan0_noise_level',
            'wlan0_rx_packets', 'wlan0_tx_packets', 'wlan0_rx_bytes', 'wlan0_tx_bytes',
            'gateway_ping_time', 'dns_resolution_time', 'memory_usage_percent', 'cpu_usage_percent'
        ]
        
        scaler_path = os.path.join(config['ai_models']['autoencoder']['model_path'], 'autoencoder_scaler.pkl')
        feature_extractor = FeatureExtractor(real_metrics, logger, scaler_path=scaler_path)
        
        # 加载测试场景
        scenarios = load_test_scenarios()
        
        # 收集原始数据和特征数据
        raw_data_list = []
        feature_data_list = []
        
        print("🔍 详细特征分析")
        print("=" * 80)
        
        for scenario in scenarios:
            raw_data = create_scenario_data(scenario['type'])
            features = feature_extractor.extract_features(raw_data)
            
            # 记录数据
            raw_data_row = {
                'scenario': scenario['name'],
                'type': scenario['type'],
                'expected': scenario['expected'],
                **raw_data
            }
            raw_data_list.append(raw_data_row)
            
            feature_data_row = {
                'scenario': scenario['name'],
                'type': scenario['type'],
                'expected': scenario['expected']
            }
            
            # 添加特征值（假设6维特征）
            for i, feature_value in enumerate(features):
                feature_data_row[f'feature_{i:02d}'] = feature_value
            
            feature_data_list.append(feature_data_row)
            
            print(f"\n📋 {scenario['name']} ({scenario['expected']})")
            print(f"   类型: {scenario['type']}")
            
            # 显示关键原始指标
            print("   原始指标:")
            for metric, value in raw_data.items():
                print(f"     {metric}: {value}")
            
            # 显示前10个特征值
            print("   提取特征 (前10个):")
            for i in range(min(10, len(features))):
                print(f"     feature_{i:02d}: {features[i]:>10.2f}")
        
        # 创建DataFrame用于分析
        raw_df = pd.DataFrame(raw_data_list)
        feature_df = pd.DataFrame(feature_data_list)
        
        # 分析正常vs异常的特征差异
        print(f"\n📊 正常 vs 异常场景特征对比")
        print("=" * 80)
        
        normal_features = feature_df[feature_df['expected'] == 'normal']
        anomaly_features = feature_df[feature_df['expected'] == 'anomaly']
        
        # 计算特征统计
        feature_cols = [col for col in feature_df.columns if col.startswith('feature_')]
        
        print("特征差异分析 (前12个特征):")
        print(f"{'特征':<12} {'正常-平均':<12} {'正常-标准差':<12} {'异常-平均':<12} {'异常-标准差':<12} {'差异倍数':<10}")
        print("-" * 80)
        
        for i, feature_col in enumerate(feature_cols[:12]):
            normal_mean = normal_features[feature_col].mean()
            normal_std = normal_features[feature_col].std()
            anomaly_mean = anomaly_features[feature_col].mean()
            anomaly_std = anomaly_features[feature_col].std()
            
            # 计算差异倍数
            if abs(normal_mean) > 1e-6:
                diff_ratio = abs(anomaly_mean - normal_mean) / abs(normal_mean)
            else:
                diff_ratio = float('inf') if abs(anomaly_mean) > 1e-6 else 0
            
            print(f"{feature_col:<12} {normal_mean:>11.2f} {normal_std:>11.2f} {anomaly_mean:>11.2f} {anomaly_std:>11.2f} {diff_ratio:>9.2f}")
        
        # 找出差异最大的特征
        print(f"\n🎯 差异最大的特征 (Top 5):")
        print("-" * 50)
        
        feature_diffs = []
        for feature_col in feature_cols:
            normal_mean = normal_features[feature_col].mean()
            anomaly_mean = anomaly_features[feature_col].mean()
            
            if abs(normal_mean) > 1e-6:
                diff_ratio = abs(anomaly_mean - normal_mean) / abs(normal_mean)
            else:
                diff_ratio = float('inf') if abs(anomaly_mean) > 1e-6 else 0
            
            feature_diffs.append((feature_col, diff_ratio, normal_mean, anomaly_mean))
        
        # 排序并显示Top 5
        feature_diffs.sort(key=lambda x: x[1], reverse=True)
        for i, (feature_col, diff_ratio, normal_mean, anomaly_mean) in enumerate(feature_diffs[:5]):
            print(f"{i+1}. {feature_col}: 差异倍数 {diff_ratio:.2f} (正常:{normal_mean:.2f}, 异常:{anomaly_mean:.2f})")
        
        # 分析问题场景
        print(f"\n⚠️  问题分析:")
        print("-" * 50)
        print("重构误差分布问题可能的原因:")
        print("1. 部分异常场景的特征变化不够极端")
        print("2. 高负载正常场景可能被错误地设计得过于极端")
        print("3. 某些异常类型在当前特征空间下难以区分")
        print("4. 需要调整异常场景的参数设置，使其更具区分性")
        
        return raw_df, feature_df
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    analyze_feature_distributions() 