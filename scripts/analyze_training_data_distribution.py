#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析训练数据的分布特征
理解自编码器学到了什么样的"正常"模式
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

def analyze_training_data():
    """分析训练数据的分布"""
    try:
        # 先尝试加载.npz文件，如果不存在则加载CSV文件
        training_data_path = os.path.join(project_root, 'data', 'training_data_v2.npz')
        
        if os.path.exists(training_data_path):
            # 加载.npz文件
            data = np.load(training_data_path)
            X = data['X']
        else:
            # 尝试加载CSV文件
            csv_path = os.path.join(project_root, 'data', 'enhanced_training_data.csv')
            if os.path.exists(csv_path):
                print(f"📁 使用CSV训练数据: {csv_path}")
                df = pd.read_csv(csv_path)
                
                # 假设前6列是特征，最后一列是标签
                if len(df.columns) >= 7:
                    X = df.iloc[:, :6].values
                else:
                    print(f"❌ CSV文件列数不足: {len(df.columns)}")
                    return None
            else:
                print(f"❌ 未找到训练数据文件")
                print(f"  尝试过的路径:")
                print(f"  - {training_data_path}")
                print(f"  - {csv_path}")
                return None
        print(f"📊 训练数据形状: {X.shape}")
        print(f"   样本数量: {X.shape[0]}")
        print(f"   特征维度: {X.shape[1]}")
        
        # 统计每个特征的分布
        print(f"\n📈 训练数据特征分布统计:")
        print("=" * 80)
        print(f"{'特征':<12} {'最小值':<12} {'最大值':<12} {'平均值':<12} {'标准差':<12} {'范围':<12}")
        print("-" * 80)
        
        feature_stats = []
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            min_val = np.min(feature_data)
            max_val = np.max(feature_data)
            mean_val = np.mean(feature_data)
            std_val = np.std(feature_data)
            range_val = max_val - min_val
            
            feature_stats.append({
                'feature': f'feature_{i:02d}',
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'range': range_val
            })
            
            print(f"feature_{i:02d}  {min_val:>11.2f} {max_val:>11.2f} {mean_val:>11.2f} {std_val:>11.2f} {range_val:>11.2f}")
        
        # 创建一些测试点来看重构误差
        print(f"\n🧪 测试不同程度的偏离对重构误差的影响:")
        print("=" * 80)
        
        # 加载配置和模型
        config_path = os.path.join(project_root, 'config', 'system_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        from src.ai_models.autoencoder_model import AutoencoderModel
        
        class SimpleLogger:
            def info(self, msg): pass
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass
            def warning(self, msg): pass
        
        logger = SimpleLogger()
        autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
        
        # 使用训练数据的平均值作为基准
        baseline = np.mean(X, axis=0)
        baseline_error = autoencoder.predict(baseline)['reconstruction_error']
        print(f"训练数据均值的重构误差: {baseline_error:.2f}")
        
        # 测试不同程度的偏离
        print(f"\n偏离程度测试:")
        print(f"{'偏离类型':<20} {'重构误差':<12} {'相对基准':<12}")
        print("-" * 50)
        
        # 1. 轻微偏离（±0.5 std）
        for factor in [0.5, 1.0, 2.0, 5.0]:
            test_point = baseline.copy()
            # 随机选择几个特征进行偏离
            selected_features = np.random.choice(X.shape[1], size=5, replace=False)
            for feat_idx in selected_features:
                std_val = feature_stats[feat_idx]['std']
                test_point[feat_idx] += factor * std_val
            
            error = autoencoder.predict(test_point)['reconstruction_error']
            relative = error / baseline_error if baseline_error > 0 else 0
            print(f"偏离{factor}倍标准差        {error:>11.2f} {relative:>11.2f}x")
        
        # 2. 极端值测试
        print(f"\n极端值测试:")
        print(f"{'极端类型':<20} {'重构误差':<12} {'相对基准':<12}")
        print("-" * 50)
        
        # 全部最小值
        extreme_min = np.array([stat['min'] for stat in feature_stats])
        error_min = autoencoder.predict(extreme_min)['reconstruction_error']
        relative_min = error_min / baseline_error if baseline_error > 0 else 0
        print(f"全部最小值             {error_min:>11.2f} {relative_min:>11.2f}x")
        
        # 全部最大值
        extreme_max = np.array([stat['max'] for stat in feature_stats])
        error_max = autoencoder.predict(extreme_max)['reconstruction_error']
        relative_max = error_max / baseline_error if baseline_error > 0 else 0
        print(f"全部最大值             {error_max:>11.2f} {relative_max:>11.2f}x")
        
        # 3. 分析我们的测试场景为什么误差低
        print(f"\n🔍 分析测试场景的特征值:")
        print("=" * 80)
        
        # 创建正常场景
        normal_data = {
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
        
        # 创建信号衰减异常场景
        signal_anomaly_data = {
            'wlan0_wireless_quality': 15.0,
            'wlan0_signal_level': -85.0,
            'wlan0_noise_level': -75.0,
            'wlan0_rx_packets': 8000,
            'wlan0_tx_packets': 5000,
            'wlan0_rx_bytes': 10000000,
            'wlan0_tx_bytes': 6000000,
            'gateway_ping_time': 150.0,
            'dns_resolution_time': 300.0,
            'memory_usage_percent': 40.0,
            'cpu_usage_percent': 25.0
        }
        
        # 初始化特征提取器
        real_metrics = list(normal_data.keys())
        scaler_path = os.path.join(config['ai_models']['autoencoder']['model_path'], 'autoencoder_scaler.pkl')
        feature_extractor = FeatureExtractor(real_metrics, logger, scaler_path=scaler_path)
        
        # 提取特征并分析
        normal_features = feature_extractor.extract_features(normal_data)
        anomaly_features = feature_extractor.extract_features(signal_anomaly_data)
        
        print(f"场景特征对比:")
        print(f"{'特征':<12} {'训练均值':<12} {'正常场景':<12} {'异常场景':<12} {'异常偏离':<12}")
        print("-" * 65)
        
        for i in range(len(normal_features)):
            training_mean = baseline[i]
            normal_val = normal_features[i]
            anomaly_val = anomaly_features[i]
            anomaly_deviation = abs(anomaly_val - training_mean) / abs(training_mean) if abs(training_mean) > 1e-6 else 0
            
            print(f"feature_{i:02d}  {training_mean:>11.2f} {normal_val:>11.2f} {anomaly_val:>11.2f} {anomaly_deviation:>11.2f}")
        
        return feature_stats, X
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    analyze_training_data() 