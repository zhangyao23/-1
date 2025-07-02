#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于实际FeatureExtractor的训练数据生成器

使用真实的特征提取器来生成训练数据，确保特征维度完全匹配。
"""

import os
import sys
import csv
import numpy as np
import random
import json
from typing import Dict, List, Any

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from feature_processor.feature_extractor import FeatureExtractor

# 定义数据目录和文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
NORMAL_TRAFFIC_FILE = os.path.join(DATA_DIR, 'normal_traffic.csv')
LABELED_ANOMALIES_FILE = os.path.join(DATA_DIR, 'labeled_anomalies.csv')

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 数据生成参数
NUM_NORMAL_SAMPLES = 18000
NUM_ANOMALY_CLUSTERS = 7
SAMPLES_PER_CLUSTER = 200
NUM_BOUNDARY_SAMPLES = 150

class SimpleLogger:
    """简单的日志记录器"""
    def info(self, msg): print(f'INFO: {msg}')
    def error(self, msg): print(f'ERROR: {msg}')
    def warning(self, msg): print(f'WARNING: {msg}')
    def debug(self, msg): pass

def load_config():
    """加载系统配置"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'system_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_base_raw_data(variation_scale: float = 1.0) -> Dict[str, float]:
    """
    生成基础的原始网络数据
    
    Args:
        variation_scale: 变化幅度缩放因子
    
    Returns:
        原始网络监控数据
    """
    base_data = {
        'wlan0_wireless_quality': 70.0 + np.random.normal(0, 15 * variation_scale),
        'wlan0_wireless_level': -55.0 + np.random.normal(0, 10 * variation_scale),
        'wlan0_packet_loss_rate': max(0, 0.01 + np.random.normal(0, 0.02 * variation_scale)),
        'wlan0_send_rate_bps': max(0, 500000.0 + np.random.normal(0, 200000 * variation_scale)),
        'wlan0_recv_rate_bps': max(0, 1500000.0 + np.random.normal(0, 500000 * variation_scale)),
        'tcp_retrans_segments': max(0, 5 + np.random.poisson(3 * variation_scale)),
        'gateway_ping_time': max(0, 12.5 + np.random.exponential(5 * variation_scale)),
        'dns_response_time': max(0, 25.0 + np.random.exponential(10 * variation_scale)),
        'tcp_connection_count': max(0, 30 + np.random.poisson(10 * variation_scale)),
        'cpu_percent': max(0, min(100, 15.0 + np.random.normal(0, 10 * variation_scale))),
        'memory_percent': max(0, min(100, 45.0 + np.random.normal(0, 15 * variation_scale)))
    }
    
    return base_data

def generate_normal_data_variations() -> List[Dict[str, float]]:
    """生成正常流量的各种变化"""
    variations = []
    
    # 主要正常模式 (80%)
    main_samples = int(NUM_NORMAL_SAMPLES * 0.8)
    for _ in range(main_samples):
        data = generate_base_raw_data(variation_scale=0.5)  # 较小变化
        variations.append(data)
    
    # 正常范围内的较大变化 (20%)
    scatter_samples = NUM_NORMAL_SAMPLES - main_samples
    for _ in range(scatter_samples):
        data = generate_base_raw_data(variation_scale=1.2)  # 较大变化但仍正常
        variations.append(data)
    
    return variations

def generate_anomaly_data_variations() -> List[tuple]:
    """生成异常数据变化，返回 (data, label) 元组列表"""
    anomaly_variations = []
    
    # 定义7种异常模式
    anomaly_patterns = [
        # 1. 信号强度异常
        lambda: {
            **generate_base_raw_data(0.3),
            'wlan0_wireless_quality': np.random.uniform(0, 30),
            'wlan0_wireless_level': np.random.uniform(-90, -80)
        },
        # 2. 高丢包率异常
        lambda: {
            **generate_base_raw_data(0.3),
            'wlan0_packet_loss_rate': np.random.uniform(0.1, 0.5),
            'tcp_retrans_segments': np.random.randint(20, 100)
        },
        # 3. 带宽异常
        lambda: {
            **generate_base_raw_data(0.3),
            'wlan0_send_rate_bps': np.random.uniform(0, 100000),
            'wlan0_recv_rate_bps': np.random.uniform(0, 200000)
        },
        # 4. 延迟异常
        lambda: {
            **generate_base_raw_data(0.3),
            'gateway_ping_time': np.random.uniform(100, 1000),
            'dns_response_time': np.random.uniform(200, 2000)
        },
        # 5. 连接数异常
        lambda: {
            **generate_base_raw_data(0.3),
            'tcp_connection_count': np.random.randint(200, 1000)
        },
        # 6. 系统资源异常
        lambda: {
            **generate_base_raw_data(0.3),
            'cpu_percent': np.random.uniform(80, 100),
            'memory_percent': np.random.uniform(85, 100)
        },
        # 7. 综合异常
        lambda: {
            **generate_base_raw_data(2.0),  # 大幅变化
            'wlan0_wireless_quality': np.random.uniform(20, 40),
            'wlan0_packet_loss_rate': np.random.uniform(0.05, 0.2),
            'gateway_ping_time': np.random.uniform(50, 200)
        }
    ]
    
    # 为每个异常模式生成数据
    for i, pattern_func in enumerate(anomaly_patterns):
        label = f'anomaly_cluster_{i+1}'
        for _ in range(SAMPLES_PER_CLUSTER):
            data = pattern_func()
            anomaly_variations.append((data, label))
    
    # 生成边界模糊的难分类点
    for _ in range(NUM_BOUNDARY_SAMPLES):
        # 在正常和异常之间的边界
        base = generate_base_raw_data(1.5)  # 中等变化
        
        # 随机选择一个异常模式作为轻微倾向
        pattern_func = random.choice(anomaly_patterns)
        anomaly_data = pattern_func()
        
        # 混合正常和异常特征
        mixed_data = {}
        for key in base.keys():
            mix_ratio = np.random.uniform(0.3, 0.7)  # 混合比例
            mixed_data[key] = mix_ratio * base[key] + (1 - mix_ratio) * anomaly_data[key]
        
        # 随机分配标签
        label = f'anomaly_cluster_{np.random.randint(1, 8)}'
        anomaly_variations.append((mixed_data, label))
    
    return anomaly_variations

def main():
    """主函数"""
    print("开始生成基于实际FeatureExtractor的训练数据...")
    
    # 初始化FeatureExtractor
    config = load_config()
    logger = SimpleLogger()
    metrics = config['data_collection']['metrics']
    extractor = FeatureExtractor(metrics, logger)
    
    # 生成正常数据
    print(f"正在生成 {NUM_NORMAL_SAMPLES} 条正常流量数据...")
    normal_raw_variations = generate_normal_data_variations()
    
    # 提取特征并保存正常数据
    normal_features = []
    feature_names = None
    
    for raw_data in normal_raw_variations:
        feature_vector = extractor.extract_features(raw_data)
        if len(feature_vector) > 0:
            normal_features.append(feature_vector)
            if feature_names is None:
                feature_names = extractor.get_feature_names()
    
    # 保存正常数据
    with open(NORMAL_TRAFFIC_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names)
        writer.writerows(normal_features)
    
    print(f"成功创建 {NORMAL_TRAFFIC_FILE}，特征维度: {len(feature_names)}")
    
    # 生成异常数据
    print("正在生成异常数据...")
    anomaly_raw_variations = generate_anomaly_data_variations()
    
    # 重置特征提取器以避免数据污染
    extractor.reset_scaler()
    
    # 先用一些正常数据重新校准scaler
    calibration_sample = normal_raw_variations[:100]
    for raw_data in calibration_sample:
        extractor.extract_features(raw_data)
    
    # 提取异常特征
    anomaly_data = []
    for raw_data, label in anomaly_raw_variations:
        feature_vector = extractor.extract_features(raw_data)
        if len(feature_vector) > 0:
            row = list(feature_vector) + [label]
            anomaly_data.append(row)
    
    # 保存异常数据
    with open(LABELED_ANOMALIES_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = feature_names + ['label']
        writer.writerow(header)
        writer.writerows(anomaly_data)
    
    total_anomalies = len(anomaly_data)
    print(f"成功创建 {LABELED_ANOMALIES_FILE}，包含 {total_anomalies} 条异常数据")
    print(f"  - {NUM_ANOMALY_CLUSTERS} 个明确异常集群，每个 {SAMPLES_PER_CLUSTER} 样本")
    print(f"  - {NUM_BOUNDARY_SAMPLES} 个边界模糊的难分类点")
    
    print("\n=== 数据生成摘要 ===")
    print(f"正常数据: {len(normal_features)} 条")
    print(f"异常数据: {total_anomalies} 条")
    print(f"特征维度: {len(feature_names)}")
    print(f"特征名称: {feature_names}")
    print("===================")
    
    print("\n现实场景数据生成完成！")

if __name__ == "__main__":
    main() 