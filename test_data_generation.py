#!/usr/bin/env python3
"""
测试数据生成脚本，验证6种异常类型的特征区分度
"""

import numpy as np
import matplotlib.pyplot as plt
from train_realistic_end_to_end_networks import generate_realistic_network_data

def test_data_generation():
    """测试数据生成并分析异常类型分布"""
    print("🔍 测试数据生成和异常类型分布")
    print("=" * 50)
    
    # 生成测试数据
    X, y_binary, y_multiclass = generate_realistic_network_data(n_samples=10000)
    
    # 分析数据分布
    print(f"\n📊 数据分布分析:")
    print(f"  总样本数: {len(X)}")
    print(f"  正常样本: {np.sum(y_binary == 0)} ({np.sum(y_binary == 0)/len(X)*100:.1f}%)")
    print(f"  异常样本: {np.sum(y_binary == 1)} ({np.sum(y_binary == 1)/len(X)*100:.1f}%)")
    
    # 分析异常类型分布
    anomaly_indices = np.where(y_binary == 1)[0]
    anomaly_multiclass = y_multiclass[anomaly_indices] - 1  # 转换为0-5索引
    
    anomaly_types = [
        "wifi_degradation",
        "network_latency", 
        "connection_instability",
        "bandwidth_congestion",
        "system_stress",
        "dns_issues"
    ]
    
    print(f"\n📈 异常类型分布:")
    for i, anomaly_type in enumerate(anomaly_types):
        count = np.sum(anomaly_multiclass == i)
        percentage = count / len(anomaly_indices) * 100
        print(f"  {anomaly_type}: {count} ({percentage:.1f}%)")
    
    # 分析每种异常类型的特征均值
    print(f"\n🔍 异常类型特征分析:")
    feature_names = [
        "wlan0_wireless_quality",
        "wlan0_signal_level", 
        "wlan0_noise_level",
        "wlan0_rx_packets",
        "wlan0_tx_packets",
        "wlan0_rx_bytes",
        "wlan0_tx_bytes",
        "gateway_ping_time",
        "dns_resolution_time",
        "memory_usage_percent",
        "cpu_usage_percent"
    ]
    
    for i, anomaly_type in enumerate(anomaly_types):
        type_indices = np.where(anomaly_multiclass == i)[0]
        if len(type_indices) > 0:
            type_data = X[anomaly_indices[type_indices]]
            print(f"\n{anomaly_type}:")
            for j, feature_name in enumerate(feature_names):
                mean_val = np.mean(type_data[:, j])
                std_val = np.std(type_data[:, j])
                print(f"  {feature_name}: {mean_val:.2f} ± {std_val:.2f}")
    
    # 检查特征区分度
    print(f"\n🎯 特征区分度分析:")
    for i in range(len(anomaly_types)):
        for j in range(i+1, len(anomaly_types)):
            type_i_indices = np.where(anomaly_multiclass == i)[0]
            type_j_indices = np.where(anomaly_multiclass == j)[0]
            
            if len(type_i_indices) > 0 and len(type_j_indices) > 0:
                type_i_data = X[anomaly_indices[type_i_indices]]
                type_j_data = X[anomaly_indices[type_j_indices]]
                
                # 计算特征差异
                differences = []
                for k in range(X.shape[1]):
                    diff = abs(np.mean(type_i_data[:, k]) - np.mean(type_j_data[:, k]))
                    differences.append(diff)
                
                max_diff_feature = feature_names[np.argmax(differences)]
                max_diff = np.max(differences)
                
                print(f"  {anomaly_types[i]} vs {anomaly_types[j]}: 最大差异特征={max_diff_feature} (差异={max_diff:.2f})")
    
    print(f"\n✅ 数据生成测试完成")

if __name__ == "__main__":
    test_data_generation() 