#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建更合理的测试场景
确保异常场景在特征空间中有足够的区分度
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_processor.feature_extractor import FeatureExtractor
from src.ai_models.autoencoder_model import AutoencoderModel

def create_improved_scenarios():
    """创建改进的测试场景"""
    scenarios = [
        # 正常场景 - 基准
        {
            "id": 1, "name": "标准正常状态", "type": "normal_baseline", "expected": "normal",
            "data": {
                'wlan0_wireless_quality': 85.0,
                'wlan0_signal_level': -45.0,
                'wlan0_noise_level': -90.0,
                'wlan0_rx_packets': 50000,
                'wlan0_tx_packets': 35000,
                'wlan0_rx_bytes': 80000000,
                'wlan0_tx_bytes': 30000000,
                'gateway_ping_time': 8.0,
                'dns_resolution_time': 15.0,
                'memory_usage_percent': 35.0,
                'cpu_usage_percent': 15.0
            }
        },
        
        # 正常场景 - 轻负载
        {
            "id": 2, "name": "轻负载正常状态", "type": "normal_light", "expected": "normal",
            "data": {
                'wlan0_wireless_quality': 88.0,
                'wlan0_signal_level': -42.0,
                'wlan0_noise_level': -92.0,
                'wlan0_rx_packets': 25000,
                'wlan0_tx_packets': 18000,
                'wlan0_rx_bytes': 40000000,
                'wlan0_tx_bytes': 15000000,
                'gateway_ping_time': 6.0,
                'dns_resolution_time': 12.0,
                'memory_usage_percent': 25.0,
                'cpu_usage_percent': 8.0
            }
        },
        
        # 正常场景 - 繁忙但正常
        {
            "id": 3, "name": "繁忙正常状态", "type": "normal_busy", "expected": "normal",
            "data": {
                'wlan0_wireless_quality': 82.0,
                'wlan0_signal_level': -48.0,
                'wlan0_noise_level': -88.0,
                'wlan0_rx_packets': 85000,
                'wlan0_tx_packets': 65000,
                'wlan0_rx_bytes': 120000000,
                'wlan0_tx_bytes': 55000000,
                'gateway_ping_time': 12.0,
                'dns_resolution_time': 20.0,
                'memory_usage_percent': 55.0,
                'cpu_usage_percent': 35.0
            }
        },
        
        # 异常场景1 - 严重信号问题（多指标异常）
        {
            "id": 4, "name": "严重信号衰减", "type": "severe_signal_degradation", "expected": "anomaly",
            "data": {
                'wlan0_wireless_quality': 15.0,  # 极低质量
                'wlan0_signal_level': -85.0,      # 极弱信号
                'wlan0_noise_level': -75.0,       # 高噪声
                'wlan0_rx_packets': 8000,         # 包量锐减
                'wlan0_tx_packets': 5000,
                'wlan0_rx_bytes': 10000000,       # 字节量锐减  
                'wlan0_tx_bytes': 6000000,
                'gateway_ping_time': 150.0,       # 延迟激增
                'dns_resolution_time': 300.0,     # DNS超慢
                'memory_usage_percent': 40.0,
                'cpu_usage_percent': 25.0
            }
        },
        
        # 异常场景2 - 网络拥塞（多指标异常）
        {
            "id": 5, "name": "严重网络拥塞", "type": "severe_network_congestion", "expected": "anomaly",
            "data": {
                'wlan0_wireless_quality': 45.0,   # 中等质量下降
                'wlan0_signal_level': -65.0,      # 信号下降
                'wlan0_noise_level': -82.0,       # 噪声增加
                'wlan0_rx_packets': 200000,       # 包量激增
                'wlan0_tx_packets': 180000,
                'wlan0_rx_bytes': 300000000,      # 字节量激增
                'wlan0_tx_bytes': 250000000,
                'gateway_ping_time': 80.0,        # 高延迟
                'dns_resolution_time': 120.0,     # DNS慢
                'memory_usage_percent': 75.0,     # 内存高
                'cpu_usage_percent': 65.0         # CPU高
            }
        },
        
        # 异常场景3 - 极端攻击（全方位异常）
        {
            "id": 6, "name": "网络攻击", "type": "network_attack", "expected": "anomaly",
            "data": {
                'wlan0_wireless_quality': 25.0,   # 质量严重下降
                'wlan0_signal_level': -75.0,      # 信号很差
                'wlan0_noise_level': -78.0,       # 高噪声
                'wlan0_rx_packets': 800000,       # 包量爆炸
                'wlan0_tx_packets': 50000,        # 发送包异常少
                'wlan0_rx_bytes': 500000000,      # 接收字节爆炸
                'wlan0_tx_bytes': 20000000,       # 发送字节很少
                'gateway_ping_time': 200.0,       # 极高延迟
                'dns_resolution_time': 8000.0,    # DNS解析超时
                'memory_usage_percent': 90.0,     # 内存爆满
                'cpu_usage_percent': 95.0         # CPU爆满
            }
        },
        
        # 异常场景4 - 硬件故障（设备层面异常）
        {
            "id": 7, "name": "硬件性能故障", "type": "hardware_failure", "expected": "anomaly",
            "data": {
                'wlan0_wireless_quality': 60.0,   # 中等下降
                'wlan0_signal_level': -55.0,      # 轻微下降
                'wlan0_noise_level': -85.0,       # 轻微噪声
                'wlan0_rx_packets': 30000,        # 处理能力下降
                'wlan0_tx_packets': 20000,
                'wlan0_rx_bytes': 35000000,       # 吞吐下降
                'wlan0_tx_bytes': 18000000,
                'gateway_ping_time': 45.0,        # 延迟增加
                'dns_resolution_time': 80.0,      # DNS变慢
                'memory_usage_percent': 85.0,     # 内存泄露
                'cpu_usage_percent': 88.0         # CPU异常高
            }
        },
        
        # 异常场景5 - 配置错误（服务层面异常）
        {
            "id": 8, "name": "配置错误异常", "type": "configuration_error", "expected": "anomaly",
            "data": {
                'wlan0_wireless_quality': 78.0,   # 质量还可以
                'wlan0_signal_level': -50.0,      # 信号正常
                'wlan0_noise_level': -89.0,       # 噪声正常
                'wlan0_rx_packets': 45000,        # 包量正常
                'wlan0_tx_packets': 35000,
                'wlan0_rx_bytes': 70000000,       # 字节量正常
                'wlan0_tx_bytes': 40000000,
                'gateway_ping_time': 35.0,        # 网关延迟异常
                'dns_resolution_time': 2000.0,    # DNS配置错误
                'memory_usage_percent': 42.0,     # 内存正常
                'cpu_usage_percent': 48.0         # CPU轻微高
            }
        }
    ]
    
    return scenarios

def test_improved_scenarios():
    """测试改进的场景"""
    try:
        # 加载配置
        config_path = os.path.join(project_root, 'config', 'system_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建简单的日志对象
        class SimpleLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass
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
        
        # 测试改进的场景
        scenarios = create_improved_scenarios()
        
        normal_errors = []
        anomaly_errors = []
        
        print("🔍 测试改进的场景设计")
        print("=" * 70)
        
        for scenario in scenarios:
            features = feature_extractor.extract_features(scenario['data'])
            result = autoencoder.predict(features)
            error = result['reconstruction_error']
            expected = scenario['expected']
            
            if expected == 'normal':
                normal_errors.append(error)
            else:
                anomaly_errors.append(error)
            
            status = "🟢 正常" if expected == 'normal' else "🔴 异常"
            print(f"{status} {scenario['name']:<20} 误差: {error:>10.2f}")
        
        # 分析改进效果
        print(f"\n📊 改进后的误差分布:")
        print("=" * 70)
        
        if normal_errors:
            print(f"正常场景 ({len(normal_errors)}个):")
            print(f"  范围: {min(normal_errors):.2f} - {max(normal_errors):.2f}")
            print(f"  平均: {np.mean(normal_errors):.2f} ± {np.std(normal_errors):.2f}")
        
        if anomaly_errors:
            print(f"\n异常场景 ({len(anomaly_errors)}个):")
            print(f"  范围: {min(anomaly_errors):.2f} - {max(anomaly_errors):.2f}")
            print(f"  平均: {np.mean(anomaly_errors):.2f} ± {np.std(anomaly_errors):.2f}")
        
        # 推荐阈值
        if normal_errors and anomaly_errors:
            max_normal = max(normal_errors)
            min_anomaly = min(anomaly_errors)
            
            print(f"\n🎯 阈值分析:")
            print(f"正常场景最大误差: {max_normal:.2f}")
            print(f"异常场景最小误差: {min_anomaly:.2f}")
            
            if max_normal < min_anomaly:
                recommended_threshold = (max_normal + min_anomaly) / 2
                print(f"✅ 完美分离！推荐阈值: {recommended_threshold:.2f}")
                
                # 更新配置文件
                config['ai_models']['autoencoder']['anomaly_threshold'] = recommended_threshold
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print(f"✅ 已更新配置文件中的阈值")
                
            else:
                overlap = max_normal - min_anomaly
                print(f"⚠️  仍有重叠: {overlap:.2f}")
                
                # 使用平衡点
                recommended_threshold = np.percentile(normal_errors, 90)
                print(f"推荐阈值: {recommended_threshold:.2f} (90%分位数)")
        
        return scenarios, normal_errors, anomaly_errors
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    test_improved_scenarios() 