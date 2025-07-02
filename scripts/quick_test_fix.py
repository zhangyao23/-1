#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速修复测试脚本

直接将11维输入数据映射为与训练数据匹配的6维特征，
避免使用FeatureExtractor导致的数据格式不匹配问题
"""

import os
import sys
import json
import numpy as np

# 将src目录添加到Python路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from anomaly_detector.anomaly_engine import AnomalyDetectionEngine
from logger.system_logger import SystemLogger
from ai_models.autoencoder_model import AutoencoderModel
from ai_models.error_classifier import ErrorClassifier

def load_config():
    """加载系统配置"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_default_inputs():
    """获取默认输入数据"""
    return {
        'wlan0_wireless_quality': 70.0, 'wlan0_wireless_level': -55.0,
        'wlan0_packet_loss_rate': 0.01, 'wlan0_send_rate_bps': 500000.0,
        'wlan0_recv_rate_bps': 1500000.0, 'tcp_retrans_segments': 5,
        'gateway_ping_time': 12.5, 'dns_response_time': 25.0,
        'tcp_connection_count': 30, 'cpu_percent': 15.0, 'memory_percent': 45.0
    }

def convert_raw_to_6d_features(raw_data):
    """
    直接将11维原始数据转换为6维特征
    映射逻辑基于训练数据的格式
    """
    # 6个核心特征名称（与训练数据一致）
    features = np.zeros(6)
    
    # 1. avg_signal_strength: 基于wireless_quality
    # 训练数据范围：70-90（正常），15-45（异常）
    features[0] = raw_data.get('wlan0_wireless_quality', 70.0)
    
    # 2. avg_data_rate: 基于传输速率，归一化到0-1
    # 训练数据范围：0.45-0.75（正常），0.1-0.5（异常）
    send_rate = raw_data.get('wlan0_send_rate_bps', 500000.0)
    recv_rate = raw_data.get('wlan0_recv_rate_bps', 1500000.0)
    avg_rate = (send_rate + recv_rate) / 2
    # 将速率归一化到0-1范围（假设最大速率为2000000 bps）
    features[1] = min(avg_rate / 2000000.0, 1.0)
    
    # 3. avg_latency: 基于网络延迟
    # 训练数据范围：10-30ms（正常），50-350ms（异常）
    ping_time = raw_data.get('gateway_ping_time', 12.5)
    dns_time = raw_data.get('dns_response_time', 25.0)
    features[2] = (ping_time + dns_time) / 2
    
    # 4. total_packet_loss: 基于丢包率和重传
    # 训练数据范围：0.001-0.05（正常），0.05-0.8（异常）
    packet_loss = raw_data.get('wlan0_packet_loss_rate', 0.01)
    retrans = raw_data.get('tcp_retrans_segments', 5)
    # 优化重传次数转换逻辑：5次重传不应等于5%丢包
    # 正常情况下5次重传约等于0.5%额外丢包
    retrans_loss = min(retrans / 1000.0, 0.05)  # 最多贡献5%丢包率
    features[3] = packet_loss + retrans_loss
    
    # 5. cpu_usage: CPU使用率
    # 训练数据范围：5-30%（正常），60-95%（异常）
    features[4] = raw_data.get('cpu_percent', 15.0)
    
    # 6. memory_usage: 内存使用率
    # 训练数据范围：30-70%（正常），65-95%（异常）
    features[5] = raw_data.get('memory_percent', 45.0)
    
    return features

def test_detection(raw_data):
    """测试异常检测"""
    print("=== 6维特征映射测试 ===\n")
    
    print("1. 原始11维输入数据:")
    for key, value in raw_data.items():
        print(f"   {key}: {value}")
    print()
    
    # 直接映射为6维特征
    features = convert_raw_to_6d_features(raw_data)
    feature_names = [
        'avg_signal_strength', 'avg_data_rate', 'avg_latency',
        'total_packet_loss', 'cpu_usage', 'memory_usage'
    ]
    
    print("2. 映射后的6维特征:")
    for i, (name, value) in enumerate(zip(feature_names, features)):
        print(f"   {name}: {value:.6f}")
    print()
    
    # 加载配置和模型
    config = load_config()
    logger = SystemLogger(config['logging'])
    logger.set_log_level('WARNING')
    
    autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
    classifier = ErrorClassifier(config['ai_models']['classifier'], logger)
    
    engine = AnomalyDetectionEngine(
        config=config['anomaly_detection'],
        autoencoder=autoencoder, error_classifier=classifier,
        buffer_manager=None, logger=logger
    )
    
    # 检测异常
    print("3. 异常检测结果:")
    is_anomaly, details = engine.detect_anomaly_from_vector(features, feature_names)
    
    print("\n" + "="*12 + " 检测结果 " + "="*12)
    if is_anomaly:
        print("\033[91m状态: 检测到异常!\033[0m")
        predicted_class = details.get('predicted_class', 'N/A')
        confidence = details.get('confidence', 0.0)
        print(f"预测类型: {predicted_class}")
        print(f"置信度: {confidence:.2%}")
    else:
        print("\033[92m状态: 一切正常\033[0m")

    print("\n--- 详细技术信息 ---")
    error = details.get('reconstruction_error', 'N/A')
    threshold = details.get('threshold', 'N/A')
    print(f"模型重构误差: {error}")
    print(f"模型异常阈值: {threshold}")
    print("="*36 + "\n")

def main():
    """主函数"""
    try:
        # 测试默认数据
        default_data = get_default_inputs()
        test_detection(default_data)
        
        # 测试几个异常场景
        print("\n=== 测试异常场景 ===\n")
        
        # 1. 信号衰减
        signal_degradation_data = default_data.copy()
        signal_degradation_data.update({
            'wlan0_wireless_quality': 25.0,  # 信号质量差
            'wlan0_send_rate_bps': 100000.0,  # 传输速率低
            'gateway_ping_time': 120.0,  # 延迟高
        })
        print("测试场景: 信号衰减")
        test_detection(signal_degradation_data)
        
        # 2. 资源过载
        resource_overload_data = default_data.copy()
        resource_overload_data.update({
            'cpu_percent': 90.0,  # CPU高
            'memory_percent': 92.0,  # 内存高
            'gateway_ping_time': 60.0,  # 延迟稍高
        })
        print("测试场景: 资源过载")
        test_detection(resource_overload_data)
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 