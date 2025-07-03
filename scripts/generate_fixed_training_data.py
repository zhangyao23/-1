#!/usr/bin/env python3
"""
Fixed training data generation script
Generates 6D feature training data without premature standardization
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_processor.feature_extractor import FeatureExtractor

# Setup simple logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic_network_metrics():
    """
    Generate realistic network metrics without premature standardization
    """
    metrics = {}
    
    # Basic network info
    metrics['interface'] = 'wlan0'
    metrics['timestamp'] = '2024-01-01 12:00:00'
    
    # Signal strength (realistic WiFi values)
    # Quality: 0-100, Level: -100 to -30 dBm
    metrics['wlan0_wireless_quality'] = np.random.uniform(30, 95)
    metrics['wlan0_wireless_level'] = np.random.uniform(-80, -35)
    
    # Data rates (realistic values in bps)
    rx_rate = np.random.uniform(500000, 50000000)  # 0.5-50 Mbps in bps
    tx_rate = np.random.uniform(500000, 30000000)  # 0.5-30 Mbps in bps
    metrics['rx_bytes_rate'] = rx_rate
    metrics['tx_bytes_rate'] = tx_rate
    
    # Latency metrics (realistic ping times in ms)
    metrics['gateway_ping_time'] = np.random.uniform(1, 50)
    metrics['dns_response_time'] = np.random.uniform(5, 100)
    
    # Packet metrics
    metrics['rx_packets_rate'] = rx_rate / np.random.uniform(800, 1500)  # Based on typical packet sizes
    metrics['tx_packets_rate'] = tx_rate / np.random.uniform(800, 1500)
    
    # Error rates (very low for normal traffic)
    metrics['rx_errors_rate'] = np.random.uniform(0, 0.01)
    metrics['tx_errors_rate'] = np.random.uniform(0, 0.01)
    
    # CPU and memory (realistic system load)
    metrics['cpu_usage'] = np.random.uniform(10, 80)
    metrics['memory_usage'] = np.random.uniform(20, 90)
    
    return metrics

def generate_anomalous_metrics(anomaly_type):
    """
    Generate metrics with specific anomaly patterns
    """
    base_metrics = generate_realistic_network_metrics()
    
    if anomaly_type == 0:  # 网络拥塞
        # 高延迟，低带宽
        base_metrics['gateway_ping_time'] = np.random.uniform(100, 500)
        base_metrics['dns_response_time'] = np.random.uniform(200, 1000)
        base_metrics['rx_bytes_rate'] = np.random.uniform(10000, 500000)  # 0.01-0.5 Mbps
        base_metrics['tx_bytes_rate'] = np.random.uniform(10000, 300000)
        
    elif anomaly_type == 1:  # 信号质量差
        # 低信号强度和质量
        base_metrics['wlan0_wireless_quality'] = np.random.uniform(5, 30)
        base_metrics['wlan0_wireless_level'] = np.random.uniform(-95, -80)
        base_metrics['rx_errors_rate'] = np.random.uniform(0.05, 0.5)
        base_metrics['tx_errors_rate'] = np.random.uniform(0.05, 0.5)
        
    elif anomaly_type == 2:  # 系统过载
        # 高CPU和内存使用
        base_metrics['cpu_usage'] = np.random.uniform(85, 100)
        base_metrics['memory_usage'] = np.random.uniform(90, 100)
        base_metrics['gateway_ping_time'] = np.random.uniform(50, 200)
        
    elif anomaly_type == 3:  # 网络攻击
        # 异常高的数据传输和包传输
        base_metrics['rx_bytes_rate'] = np.random.uniform(80000000, 200000000)  # 80-200 Mbps
        base_metrics['tx_bytes_rate'] = np.random.uniform(50000000, 150000000)
        base_metrics['rx_packets_rate'] = np.random.uniform(10000, 50000)
        base_metrics['tx_packets_rate'] = np.random.uniform(10000, 50000)
        
    elif anomaly_type == 4:  # 硬件故障
        # 高错误率
        base_metrics['rx_errors_rate'] = np.random.uniform(0.1, 1.0)
        base_metrics['tx_errors_rate'] = np.random.uniform(0.1, 1.0)
        base_metrics['wlan0_wireless_quality'] = np.random.uniform(0, 20)
        
    elif anomaly_type == 5:  # 配置错误
        # 不一致的网络参数
        base_metrics['dns_response_time'] = np.random.uniform(500, 2000)
        base_metrics['gateway_ping_time'] = np.random.uniform(200, 800)
        # 数据率不匹配
        ratio = np.random.uniform(10, 100)  # 异常的上下行比例
        base_metrics['tx_bytes_rate'] = base_metrics['rx_bytes_rate'] / ratio
    
    return base_metrics

def main():
    print("开始生成修复后的训练数据...")
    
    # Initialize feature extractor with required parameters
    metrics_config = [
        'wlan0_wireless_quality', 'wlan0_wireless_level',
        'rx_bytes_rate', 'tx_bytes_rate', 
        'gateway_ping_time', 'dns_response_time',
        'rx_packets_rate', 'tx_packets_rate',
        'rx_errors_rate', 'tx_errors_rate',
        'cpu_usage', 'memory_usage'
    ]
    feature_extractor = FeatureExtractor(metrics_config, logger, scaler_path=None)
    
    # Generate normal traffic data
    print("生成正常流量数据...")
    normal_features = []
    normal_labels = []
    
    for i in range(15000):
        if i % 1000 == 0:
            print(f"正常数据进度: {i}/15000")
        
        metrics = generate_realistic_network_metrics()
        features = feature_extractor.extract_features(metrics)
        
        if features is not None and len(features) == 6:
            normal_features.append(features)
            normal_labels.append(0)  # 0 = normal
    
    # Generate anomalous data
    print("生成异常流量数据...")
    anomaly_features = []
    anomaly_labels = []
    
    # 300 samples per anomaly type
    for anomaly_type in range(6):
        print(f"生成异常类型 {anomaly_type} 的数据...")
        for i in range(300):
            metrics = generate_anomalous_metrics(anomaly_type)
            features = feature_extractor.extract_features(metrics)
            
            if features is not None and len(features) == 6:
                anomaly_features.append(features)
                anomaly_labels.append(anomaly_type + 1)  # 1-6 for different anomaly types
    
    # Combine all data
    all_features = normal_features + anomaly_features
    all_labels = normal_labels + anomaly_labels
    
    # Create DataFrame
    feature_names = [
        'avg_signal_strength',
        'avg_data_rate',
        'avg_latency', 
        'packet_loss_rate',
        'system_load',
        'network_stability'
    ]
    
    df = pd.DataFrame(all_features, columns=feature_names)
    df['label'] = all_labels
    
    # Create anomaly type mapping
    anomaly_types = {
        0: 'normal',
        1: 'network_congestion', 
        2: 'poor_signal_quality',
        3: 'system_overload',
        4: 'network_attack',
        5: 'hardware_failure',
        6: 'configuration_error'
    }
    
    df['anomaly_type'] = df['label'].map(anomaly_types)
    
    # Save data
    os.makedirs('data', exist_ok=True)
    output_file = 'data/fixed_training_data_6d.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n训练数据已保存到: {output_file}")
    print(f"总样本数: {len(df)}")
    print(f"正常样本数: {len(normal_features)}")
    print(f"异常样本数: {len(anomaly_features)}")
    
    # Print feature statistics
    print("\n特征统计信息:")
    print(df[feature_names].describe())
    
    # Print label distribution
    print("\n标签分布:")
    print(df['anomaly_type'].value_counts())
    
    print("\n修复后的训练数据生成完成！")

if __name__ == "__main__":
    main() 