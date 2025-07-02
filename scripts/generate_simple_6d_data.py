#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化的6维特征数据生成器

直接生成6维特征数据，不通过FeatureExtractor进行处理
基于真实网络性能指标的正常和异常模式
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# 配置参数
NUM_NORMAL_SAMPLES = 15000
NUM_ANOMALY_SAMPLES_PER_TYPE = 300

# 6个核心特征名称（与FeatureExtractor._convert_to_vector中的顺序一致）
FEATURE_NAMES = [
    'avg_signal_strength',  # 平均信号强度
    'avg_data_rate',        # 平均数据传输速率
    'avg_latency',          # 平均网络延迟
    'total_packet_loss',    # 总丢包率
    'cpu_usage',            # CPU使用率
    'memory_usage'          # 内存使用率
]

# 异常类型
ANOMALY_TYPES = [
    'signal_degradation',   # 信号衰减
    'network_congestion',   # 网络拥塞
    'connection_timeout',   # 连接超时
    'packet_corruption',    # 数据包损坏
    'resource_overload',    # 资源过载
    'mixed_anomaly'         # 混合异常
]

def generate_normal_features():
    """生成正常的6维特征数据"""
    # 基于真实网络性能指标的正常范围
    features = np.random.normal(size=(NUM_NORMAL_SAMPLES, 6))
    
    # 为每个特征设置合理的均值和标准差
    # avg_signal_strength: 信号强度 (70-90)
    features[:, 0] = features[:, 0] * 5 + 80
    
    # avg_data_rate: 数据传输速率 (正常化到0-1范围)
    features[:, 1] = (features[:, 1] * 0.15 + 0.6)  # 0.45-0.75左右
    
    # avg_latency: 网络延迟 (10-30ms)
    features[:, 2] = np.abs(features[:, 2]) * 8 + 15
    
    # total_packet_loss: 丢包率 (0.001-0.05)
    features[:, 3] = np.abs(features[:, 3]) * 0.02 + 0.01
    
    # cpu_usage: CPU使用率 (5-30%)
    features[:, 4] = np.abs(features[:, 4]) * 8 + 15
    
    # memory_usage: 内存使用率 (30-70%)
    features[:, 5] = np.abs(features[:, 5]) * 12 + 50
    
    return features

def generate_anomaly_features(anomaly_type, num_samples):
    """生成指定类型的异常特征数据"""
    features = np.random.normal(size=(num_samples, 6))
    
    if anomaly_type == 'signal_degradation':
        # 信号衰减：信号强度低，传输速率下降，延迟增加
        features[:, 0] = features[:, 0] * 8 + 30   # 信号强度 15-45
        features[:, 1] = features[:, 1] * 0.1 + 0.2  # 传输速率低 0.1-0.3
        features[:, 2] = np.abs(features[:, 2]) * 30 + 80  # 延迟高 50-140ms
        features[:, 3] = np.abs(features[:, 3]) * 0.1 + 0.15  # 丢包率高
        features[:, 4] = np.abs(features[:, 4]) * 8 + 15   # CPU正常
        features[:, 5] = np.abs(features[:, 5]) * 12 + 50  # 内存正常
        
    elif anomaly_type == 'network_congestion':
        # 网络拥塞：高丢包率，高延迟，传输速率下降
        features[:, 0] = features[:, 0] * 6 + 55   # 信号强度中等 45-65
        features[:, 1] = features[:, 1] * 0.12 + 0.25  # 传输速率低
        features[:, 2] = np.abs(features[:, 2]) * 25 + 90  # 延迟高 65-140ms
        features[:, 3] = np.abs(features[:, 3]) * 0.15 + 0.2  # 丢包率很高
        features[:, 4] = np.abs(features[:, 4]) * 10 + 20  # CPU稍高
        features[:, 5] = np.abs(features[:, 5]) * 12 + 50  # 内存正常
        
    elif anomaly_type == 'connection_timeout':
        # 连接超时：极高延迟，传输速率严重下降
        features[:, 0] = features[:, 0] * 8 + 60   # 信号强度中等 50-70
        features[:, 1] = features[:, 1] * 0.08 + 0.1  # 传输速率极低
        features[:, 2] = np.abs(features[:, 2]) * 50 + 200  # 延迟极高 150-350ms
        features[:, 3] = np.abs(features[:, 3]) * 0.2 + 0.3  # 丢包率极高
        features[:, 4] = np.abs(features[:, 4]) * 8 + 15   # CPU正常
        features[:, 5] = np.abs(features[:, 5]) * 12 + 50  # 内存正常
        
    elif anomaly_type == 'packet_corruption':
        # 数据包损坏：中等丢包率，传输受影响
        features[:, 0] = features[:, 0] * 10 + 50  # 信号强度中等 40-60
        features[:, 1] = features[:, 1] * 0.15 + 0.35  # 传输速率中等 0.2-0.5
        features[:, 2] = np.abs(features[:, 2]) * 20 + 40  # 延迟中等 20-80ms
        features[:, 3] = np.abs(features[:, 3]) * 0.08 + 0.08  # 丢包率中高
        features[:, 4] = np.abs(features[:, 4]) * 8 + 15   # CPU正常
        features[:, 5] = np.abs(features[:, 5]) * 12 + 50  # 内存正常
        
    elif anomaly_type == 'resource_overload':
        # 资源过载：CPU和内存使用率极高
        features[:, 0] = features[:, 0] * 6 + 70   # 信号强度正常 60-80
        features[:, 1] = features[:, 1] * 0.15 + 0.45  # 传输速率稍低
        features[:, 2] = np.abs(features[:, 2]) * 15 + 35  # 延迟稍高 20-65ms
        features[:, 3] = np.abs(features[:, 3]) * 0.03 + 0.02  # 丢包率稍高
        features[:, 4] = np.abs(features[:, 4]) * 8 + 85   # CPU极高 80-95%
        features[:, 5] = np.abs(features[:, 5]) * 8 + 88   # 内存极高 85-95%
        
    elif anomaly_type == 'mixed_anomaly':
        # 混合异常：多个指标同时异常
        features[:, 0] = features[:, 0] * 10 + 35  # 信号强度低 25-45
        features[:, 1] = features[:, 1] * 0.12 + 0.2  # 传输速率低
        features[:, 2] = np.abs(features[:, 2]) * 40 + 120  # 延迟高 80-200ms
        features[:, 3] = np.abs(features[:, 3]) * 0.12 + 0.15  # 丢包率高
        features[:, 4] = np.abs(features[:, 4]) * 15 + 70  # CPU高 60-85%
        features[:, 5] = np.abs(features[:, 5]) * 15 + 75  # 内存高 65-90%
    
    # 确保所有值在合理范围内
    features[:, 0] = np.clip(features[:, 0], 10, 100)  # 信号强度
    features[:, 1] = np.clip(features[:, 1], 0.05, 1.0)  # 传输速率
    features[:, 2] = np.clip(features[:, 2], 5, 500)     # 延迟
    features[:, 3] = np.clip(features[:, 3], 0.001, 0.8) # 丢包率
    features[:, 4] = np.clip(features[:, 4], 5, 100)     # CPU使用率
    features[:, 5] = np.clip(features[:, 5], 20, 100)    # 内存使用率
    
    return features

def main():
    """主函数"""
    print("🚀 开始生成简化6维特征数据...")
    
    # 创建data目录
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # 生成正常数据
    print(f"📊 生成 {NUM_NORMAL_SAMPLES} 条正常数据...")
    normal_features = generate_normal_features()
    normal_df = pd.DataFrame(normal_features, columns=FEATURE_NAMES)
    
    # 保存正常数据
    normal_file = data_dir / '6d_normal_traffic.csv'
    normal_df.to_csv(normal_file, index=False)
    print(f"✅ 正常数据已保存到: {normal_file}")
    
    # 生成异常数据
    print(f"📊 生成异常数据，每种类型 {NUM_ANOMALY_SAMPLES_PER_TYPE} 条...")
    anomaly_data = []
    anomaly_labels = []
    
    for anomaly_type in ANOMALY_TYPES:
        print(f"   - 生成 {anomaly_type} 数据...")
        anomaly_features = generate_anomaly_features(anomaly_type, NUM_ANOMALY_SAMPLES_PER_TYPE)
        anomaly_data.append(anomaly_features)
        anomaly_labels.extend([anomaly_type] * NUM_ANOMALY_SAMPLES_PER_TYPE)
    
    # 合并所有异常数据
    all_anomaly_features = np.vstack(anomaly_data)
    anomaly_df = pd.DataFrame(all_anomaly_features, columns=FEATURE_NAMES)
    anomaly_df['anomaly_type'] = anomaly_labels
    
    # 保存异常数据
    anomaly_file = data_dir / '6d_labeled_anomalies.csv'
    anomaly_df.to_csv(anomaly_file, index=False)
    print(f"✅ 异常数据已保存到: {anomaly_file}")
    
    # 打印统计信息
    print(f"\n📈 数据生成摘要:")
    print(f"正常数据: {len(normal_df)} 条")
    print(f"异常数据: {len(anomaly_df)} 条")
    print(f"特征维度: {len(FEATURE_NAMES)}")
    print(f"异常类型: {len(ANOMALY_TYPES)} 种")
    
    print(f"\n📊 正常数据统计:")
    print(normal_df.describe())
    
    print(f"\n📊 异常数据统计:")
    print(anomaly_df.groupby('anomaly_type').size())
    
    print(f"\n🎯 6维特征数据生成完成!")
    print(f"下一步：重新训练模型")
    print(f"  python3 scripts/train_model.py autoencoder --data_path {normal_file}")
    print(f"  python3 scripts/train_model.py classifier --data_path {anomaly_file}")

if __name__ == "__main__":
    main() 