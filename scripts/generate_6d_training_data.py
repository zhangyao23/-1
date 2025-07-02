#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6维特征训练数据生成器

基于新的6维特征架构生成训练数据：
- 输入：11个真实网络指标
- 特征工程：11维 → 6维特征向量（降维处理）  
- 异常类型：6种错误类型
- 使用FeatureExtractor确保特征格式一致性

异常类型：
1. signal_degradation - 信号衰减
2. network_congestion - 网络拥塞  
3. connection_timeout - 连接超时
4. packet_corruption - 数据包损坏
5. resource_overload - 资源过载
6. mixed_anomaly - 混合异常
"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_processor.feature_extractor import FeatureExtractor
from src.logger.system_logger import SystemLogger

# 配置参数
NUM_NORMAL_SAMPLES = 15000
NUM_ANOMALY_SAMPLES_PER_TYPE = 300  # 每种异常类型300个样本

# 文件路径
DATA_DIR = os.path.join(project_root, 'data')
NORMAL_TRAFFIC_FILE = os.path.join(DATA_DIR, '6d_normal_traffic.csv')
LABELED_ANOMALIES_FILE = os.path.join(DATA_DIR, '6d_labeled_anomalies.csv')
CONFIG_FILE = os.path.join(project_root, 'config', 'system_config.json')

class SimpleLogger:
    """简单日志类"""
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def debug(self, msg): pass
    def warning(self, msg): print(f"WARNING: {msg}")

def load_config():
    """加载配置文件"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_normal_network_data() -> Dict[str, float]:
    """生成正常网络数据 - 使用正确的字段名"""
    return {
        # WiFi信号质量指标
        'wlan0_wireless_quality': np.random.uniform(65, 95),
        'wlan0_wireless_level': np.random.uniform(-65, -35),
        
        # 网络传输速率
        'wlan0_send_rate_bps': np.random.uniform(400000, 800000),
        'wlan0_recv_rate_bps': np.random.uniform(1000000, 2000000),
        
        # 数据包丢失和重传
        'wlan0_packet_loss_rate': np.random.uniform(0.001, 0.05),
        'tcp_retrans_segments': np.random.uniform(1, 10),
        
        # 网络延迟
        'gateway_ping_time': np.random.uniform(5, 25),
        'dns_response_time': np.random.uniform(15, 60),
        
        # 连接数
        'tcp_connection_count': np.random.uniform(20, 50),
        
        # 系统资源
        'cpu_percent': np.random.uniform(5, 25),
        'memory_percent': np.random.uniform(30, 65)
    }

def generate_signal_degradation_data() -> Dict[str, float]:
    """生成信号衰减异常数据"""
    data = generate_normal_network_data()
    
    # 信号质量严重下降
    data['wlan0_wireless_quality'] = np.random.uniform(15, 35)
    data['wlan0_wireless_level'] = np.random.uniform(-95, -75)
    
    # 传输速率严重下降
    data['wlan0_send_rate_bps'] = np.random.uniform(50000, 200000)
    data['wlan0_recv_rate_bps'] = np.random.uniform(100000, 500000)
    
    # 丢包率增加
    data['wlan0_packet_loss_rate'] = np.random.uniform(0.1, 0.3)
    data['tcp_retrans_segments'] = np.random.uniform(20, 50)
    
    # 延迟大幅增加
    data['gateway_ping_time'] = np.random.uniform(80, 200)
    data['dns_response_time'] = np.random.uniform(150, 400)
    
    # 连接数可能减少
    data['tcp_connection_count'] = np.random.uniform(5, 15)
    
    return data

def generate_network_congestion_data() -> Dict[str, float]:
    """生成网络拥塞异常数据"""
    data = generate_normal_network_data()
    
    # 信号质量中等，但传输受限
    data['wlan0_wireless_quality'] = np.random.uniform(45, 70)
    data['wlan0_wireless_level'] = np.random.uniform(-70, -50)
    
    # 传输速率严重下降
    data['wlan0_send_rate_bps'] = np.random.uniform(100000, 300000)
    data['wlan0_recv_rate_bps'] = np.random.uniform(200000, 800000)
    
    # 高丢包率和重传
    data['wlan0_packet_loss_rate'] = np.random.uniform(0.08, 0.25)
    data['tcp_retrans_segments'] = np.random.uniform(15, 40)
    
    # 延迟显著增加
    data['gateway_ping_time'] = np.random.uniform(60, 150)
    data['dns_response_time'] = np.random.uniform(100, 300)
    
    # 连接数增加（拥塞原因）
    data['tcp_connection_count'] = np.random.uniform(80, 150)
    
    return data

def generate_connection_timeout_data() -> Dict[str, float]:
    """生成连接超时异常数据"""
    data = generate_normal_network_data()
    
    # 信号质量正常，但连接不稳定
    data['wlan0_wireless_quality'] = np.random.uniform(55, 80)
    data['wlan0_wireless_level'] = np.random.uniform(-65, -40)
    
    # 传输速率大幅下降
    data['wlan0_send_rate_bps'] = np.random.uniform(10000, 100000)
    data['wlan0_recv_rate_bps'] = np.random.uniform(50000, 300000)
    
    # 极高丢包率
    data['wlan0_packet_loss_rate'] = np.random.uniform(0.2, 0.5)
    data['tcp_retrans_segments'] = np.random.uniform(30, 80)
    
    # 极高延迟或超时
    data['gateway_ping_time'] = np.random.uniform(300, 800)
    data['dns_response_time'] = np.random.uniform(500, 1200)
    
    # 连接数减少（超时导致）
    data['tcp_connection_count'] = np.random.uniform(2, 10)
    
    return data

def generate_packet_corruption_data() -> Dict[str, float]:
    """生成数据包损坏异常数据"""
    data = generate_normal_network_data()
    
    # 信号质量中等
    data['wlan0_wireless_quality'] = np.random.uniform(35, 65)
    data['wlan0_wireless_level'] = np.random.uniform(-80, -50)
    
    # 传输速率受影响
    data['wlan0_send_rate_bps'] = np.random.uniform(150000, 400000)
    data['wlan0_recv_rate_bps'] = np.random.uniform(300000, 900000)
    
    # 高丢包率和重传（包损坏导致）
    data['wlan0_packet_loss_rate'] = np.random.uniform(0.05, 0.15)
    data['tcp_retrans_segments'] = np.random.uniform(25, 60)
    
    # 延迟不稳定
    data['gateway_ping_time'] = np.random.uniform(35, 90)
    data['dns_response_time'] = np.random.uniform(70, 200)
    
    # 连接数可能正常
    data['tcp_connection_count'] = np.random.uniform(25, 45)
    
    return data

def generate_resource_overload_data() -> Dict[str, float]:
    """生成资源过载异常数据"""
    data = generate_normal_network_data()
    
    # 网络指标可能正常
    data['wlan0_wireless_quality'] = np.random.uniform(60, 85)
    data['wlan0_wireless_level'] = np.random.uniform(-60, -40)
    
    # 传输速率轻微受影响
    data['wlan0_send_rate_bps'] = np.random.uniform(300000, 600000)
    data['wlan0_recv_rate_bps'] = np.random.uniform(800000, 1500000)
    
    # 轻微网络延迟（资源不足导致）
    data['gateway_ping_time'] = np.random.uniform(30, 80)
    data['dns_response_time'] = np.random.uniform(60, 150)
    
    # 系统资源严重不足
    data['cpu_percent'] = np.random.uniform(80, 98)
    data['memory_percent'] = np.random.uniform(85, 99)
    
    # 连接数可能增加
    data['tcp_connection_count'] = np.random.uniform(60, 100)
    
    return data

def generate_mixed_anomaly_data() -> Dict[str, float]:
    """生成混合异常数据"""
    data = generate_normal_network_data()
    
    # 多个问题同时出现
    data['wlan0_wireless_quality'] = np.random.uniform(20, 50)
    data['wlan0_wireless_level'] = np.random.uniform(-85, -65)
    
    # 传输速率严重受影响
    data['wlan0_send_rate_bps'] = np.random.uniform(50000, 250000)
    data['wlan0_recv_rate_bps'] = np.random.uniform(150000, 600000)
    
    # 高丢包率
    data['wlan0_packet_loss_rate'] = np.random.uniform(0.1, 0.4)
    data['tcp_retrans_segments'] = np.random.uniform(40, 100)
    
    # 高延迟
    data['gateway_ping_time'] = np.random.uniform(100, 300)
    data['dns_response_time'] = np.random.uniform(200, 500)
    
    # 系统资源问题
    data['cpu_percent'] = np.random.uniform(70, 95)
    data['memory_percent'] = np.random.uniform(75, 95)
    
    # 连接数异常
    data['tcp_connection_count'] = np.random.uniform(5, 120)
    
    return data

def generate_anomaly_variations() -> List[Tuple[Dict[str, float], str]]:
    """生成所有类型的异常数据变体"""
    anomaly_generators = {
        'signal_degradation': generate_signal_degradation_data,
        'network_congestion': generate_network_congestion_data,
        'connection_timeout': generate_connection_timeout_data,
        'packet_corruption': generate_packet_corruption_data,
        'resource_overload': generate_resource_overload_data,
        'mixed_anomaly': generate_mixed_anomaly_data
    }
    
    anomaly_data = []
    
    for anomaly_type, generator in anomaly_generators.items():
        print(f"生成 {anomaly_type} 异常数据...")
        for _ in range(NUM_ANOMALY_SAMPLES_PER_TYPE):
            data = generator()
            anomaly_data.append((data, anomaly_type))
    
    return anomaly_data

def main():
    """主函数"""
    print("🚀 开始生成6维特征训练数据")
    print("="*60)
    
    # 加载配置
    config = load_config()
    logger = SimpleLogger()
    
    # 创建特征提取器（不使用预训练的scaler）
    print("🔧 初始化特征提取器...")
    real_metrics = [
        'wlan0_wireless_quality', 'wlan0_signal_level', 'wlan0_noise_level',
        'wlan0_rx_packets', 'wlan0_tx_packets', 'wlan0_rx_bytes', 'wlan0_tx_bytes',
        'gateway_ping_time', 'dns_resolution_time', 'memory_usage_percent', 'cpu_usage_percent'
    ]
    
    feature_extractor = FeatureExtractor(
        metrics_config=real_metrics,
        logger=logger,
        scaler_path=None  # 训练模式，不使用预训练scaler
    )
    
    # 生成正常数据
    print(f"🔄 生成 {NUM_NORMAL_SAMPLES} 条正常数据...")
    normal_features = []
    feature_names = None
    
    for i in range(NUM_NORMAL_SAMPLES):
        if (i + 1) % 2000 == 0:
            print(f"   进度: {i + 1}/{NUM_NORMAL_SAMPLES}")
        
        raw_data = generate_normal_network_data()
        feature_vector = feature_extractor.extract_features(raw_data)
        
        if len(feature_vector) > 0:
            normal_features.append(feature_vector)
            if feature_names is None:
                feature_names = feature_extractor.get_feature_names()
    
    print(f"✅ 正常数据生成完成: {len(normal_features)} 条，{len(feature_names)} 维特征")
    
    # 保存正常数据
    normal_df = pd.DataFrame(normal_features, columns=feature_names)
    normal_df.to_csv(NORMAL_TRAFFIC_FILE, index=False)
    print(f"📁 正常数据已保存到: {NORMAL_TRAFFIC_FILE}")
    
    # 重置特征提取器的scaler，准备处理异常数据
    feature_extractor.reset_scaler()
    
    # 用一些正常数据重新校准scaler
    print("🔄 使用正常数据校准特征提取器...")
    calibration_samples = normal_features[:500]  # 使用前500个样本校准
    for sample in calibration_samples:
        # 这里需要将特征向量转换回原始数据格式进行校准
        # 为简化，我们使用新的正常数据
        calibration_raw = generate_normal_network_data()
        feature_extractor.extract_features(calibration_raw)
    
    # 生成异常数据
    print("🔄 生成异常数据...")
    anomaly_variations = generate_anomaly_variations()
    
    anomaly_features = []
    anomaly_labels = []
    
    for raw_data, label in anomaly_variations:
        feature_vector = feature_extractor.extract_features(raw_data)
        if len(feature_vector) > 0:
            anomaly_features.append(feature_vector)
            anomaly_labels.append(label)
    
    print(f"✅ 异常数据生成完成: {len(anomaly_features)} 条")
    
    # 保存异常数据
    anomaly_df = pd.DataFrame(anomaly_features, columns=feature_names)
    anomaly_df['label'] = anomaly_labels
    anomaly_df.to_csv(LABELED_ANOMALIES_FILE, index=False)
    print(f"📁 异常数据已保存到: {LABELED_ANOMALIES_FILE}")
    
    # 统计信息
    print(f"\n📊 数据生成摘要")
    print("="*40)
    print(f"正常数据: {len(normal_features)} 条")
    print(f"异常数据: {len(anomaly_features)} 条")
    print(f"特征维度: {len(feature_names)}")
    print(f"特征名称: {feature_names}")
    
    # 异常类型分布
    label_counts = pd.Series(anomaly_labels).value_counts()
    print(f"\n异常类型分布:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} 条")
    
    print(f"\n🎯 6维特征训练数据生成完成!")
    print("新架构特点:")
    print("✅ 降维处理：11个原始指标 → 6个核心特征")
    print("✅ 6种异常类型，每种300个样本")
    print("✅ 特征格式与实际测试环境一致")
    print("✅ 专注于最重要的6个网络性能指标")
    
    print(f"\n📋 下一步：使用新数据重新训练模型")
    print(f"  python scripts/train_model.py autoencoder --data_path {NORMAL_TRAFFIC_FILE}")
    print(f"  python scripts/train_model.py classifier --data_path {LABELED_ANOMALIES_FILE}")

if __name__ == "__main__":
    main() 