#!/usr/bin/env python3
"""
改进的6维训练数据生成器
确保所有6个特征都有合理的变化范围和真实的相关性
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple

def generate_realistic_6d_features(num_samples: int, anomaly_type: str = "normal") -> np.ndarray:
    """
    生成具有真实变化的6维特征数据
    """
    features = np.zeros((num_samples, 6))
    
    for i in range(num_samples):
        if anomaly_type == "normal":
            # 正常数据范围 - 确保所有特征都有变化
            signal_strength = np.random.normal(7.0, 1.5)  # 4-10范围
            data_rate = np.random.normal(2.5, 0.8)        # 1-4范围
            latency = np.random.normal(15.0, 5.0)         # 5-25范围
            packet_loss = np.random.normal(0.02, 0.01)    # 0-0.05范围
            system_load = np.random.normal(0.3, 0.15)     # 0-0.6范围
            network_stability = np.random.normal(0.85, 0.1) # 0.6-1.0范围
            
        elif anomaly_type == "signal_degradation":
            # 信号质量异常
            signal_strength = np.random.normal(3.0, 1.0)  # 信号弱
            data_rate = np.random.normal(1.5, 0.5)        # 数据率低
            latency = np.random.normal(25.0, 8.0)         # 延迟高
            packet_loss = np.random.normal(0.08, 0.03)    # 丢包多
            system_load = np.random.normal(0.3, 0.15)     # 正常
            network_stability = np.random.normal(0.6, 0.15) # 稳定性差
            
        elif anomaly_type == "network_congestion":
            # 网络拥堵
            signal_strength = np.random.normal(6.5, 1.0)  # 信号正常
            data_rate = np.random.normal(1.2, 0.4)        # 数据率极低
            latency = np.random.normal(40.0, 15.0)        # 延迟极高
            packet_loss = np.random.normal(0.12, 0.05)    # 丢包严重
            system_load = np.random.normal(0.7, 0.2)      # 负载高
            network_stability = np.random.normal(0.4, 0.2) # 很不稳定
            
        elif anomaly_type == "connection_timeout":
            # 连接超时
            signal_strength = np.random.normal(5.0, 2.0)  # 信号不稳定
            data_rate = np.random.normal(0.8, 0.3)        # 数据率很低
            latency = np.random.normal(60.0, 20.0)        # 延迟很高
            packet_loss = np.random.normal(0.15, 0.08)    # 严重丢包
            system_load = np.random.normal(0.4, 0.2)      # 中等负载
            network_stability = np.random.normal(0.3, 0.15) # 极不稳定
            
        elif anomaly_type == "packet_corruption":
            # 数据包损坏
            signal_strength = np.random.normal(6.0, 1.5)  # 信号中等
            data_rate = np.random.normal(2.0, 0.6)        # 数据率中等
            latency = np.random.normal(20.0, 6.0)         # 延迟中等
            packet_loss = np.random.normal(0.25, 0.1)     # 极高丢包率
            system_load = np.random.normal(0.5, 0.2)      # 中高负载
            network_stability = np.random.normal(0.5, 0.2) # 中等稳定性
            
        elif anomaly_type == "resource_overload":
            # 系统资源过载
            signal_strength = np.random.normal(6.5, 1.0)  # 信号正常
            data_rate = np.random.normal(1.8, 0.5)        # 数据率偏低
            latency = np.random.normal(35.0, 12.0)        # 延迟高
            packet_loss = np.random.normal(0.06, 0.03)    # 丢包偏多
            system_load = np.random.normal(0.9, 0.1)      # 极高负载
            network_stability = np.random.normal(0.65, 0.2) # 稳定性差
            
        elif anomaly_type == "mixed_anomaly":
            # 混合异常
            signal_strength = np.random.normal(4.5, 2.0)  # 信号差
            data_rate = np.random.normal(1.0, 0.4)        # 数据率低
            latency = np.random.normal(45.0, 18.0)        # 延迟很高
            packet_loss = np.random.normal(0.18, 0.08)    # 严重丢包
            system_load = np.random.normal(0.8, 0.2)      # 高负载
            network_stability = np.random.normal(0.35, 0.2) # 很不稳定
        
        # 应用约束确保数值在合理范围内
        signal_strength = np.clip(signal_strength, 0.5, 10.0)
        data_rate = np.clip(data_rate, 0.1, 5.0)
        latency = np.clip(latency, 1.0, 100.0)
        packet_loss = np.clip(packet_loss, 0.0, 0.5)
        system_load = np.clip(system_load, 0.0, 1.0)
        network_stability = np.clip(network_stability, 0.0, 1.0)
        
        features[i] = [signal_strength, data_rate, latency, packet_loss, system_load, network_stability]
    
    return features

def generate_improved_training_data():
    """生成改进的6维训练数据"""
    
    print("🚀 开始生成改进的6维训练数据...")
    
    # 生成正常数据
    print("📊 生成正常数据...")
    normal_features = generate_realistic_6d_features(15000, "normal")
    normal_labels = np.zeros(15000)
    normal_types = ["normal"] * 15000
    
    # 生成异常数据
    anomaly_types = [
        "signal_degradation",
        "network_congestion", 
        "connection_timeout",
        "packet_corruption",
        "resource_overload",
        "mixed_anomaly"
    ]
    
    all_anomaly_features = []
    all_anomaly_labels = []
    all_anomaly_types = []
    
    for i, anomaly_type in enumerate(anomaly_types):
        print(f"📊 生成异常数据: {anomaly_type}...")
        features = generate_realistic_6d_features(300, anomaly_type)
        labels = np.full(300, i + 1)  # 标签1-6
        types = [anomaly_type] * 300
        
        all_anomaly_features.append(features)
        all_anomaly_labels.append(labels)
        all_anomaly_types.extend(types)
    
    # 合并所有异常数据
    anomaly_features = np.vstack(all_anomaly_features)
    anomaly_labels = np.concatenate(all_anomaly_labels)
    
    # 合并正常和异常数据
    all_features = np.vstack([normal_features, anomaly_features])
    all_labels = np.concatenate([normal_labels, anomaly_labels])
    all_types = normal_types + all_anomaly_types
    
    # 创建DataFrame
    feature_names = [
        'avg_signal_strength',
        'avg_data_rate', 
        'avg_latency',
        'packet_loss_rate',
        'system_load',
        'network_stability'
    ]
    
    df = pd.DataFrame(all_features, columns=feature_names)
    df['label'] = all_labels.astype(int)
    df['anomaly_type'] = all_types
    
    # 随机打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存数据
    output_file = 'data/improved_training_data_6d.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ 改进的训练数据已保存: {output_file}")
    print(f"📊 数据统计:")
    print(f"   - 总样本数: {len(df)}")
    print(f"   - 正常样本: {len(df[df['label'] == 0])}")
    print(f"   - 异常样本: {len(df[df['label'] > 0])}")
    print()
    
    # 显示特征统计
    print("📈 特征统计信息:")
    for col in feature_names:
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"   {col}: 均值={mean_val:.3f}, 标准差={std_val:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]")
    
    return output_file

if __name__ == "__main__":
    generate_improved_training_data() 