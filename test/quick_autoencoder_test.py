#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自编码器异常检测快速测试
验证重构误差和异常检测阈值的正确性
"""

import sys
import pandas as pd
import numpy as np

sys.path.append('src')
sys.path.append('.')

print('🤖 自编码器异常检测快速测试')
print('=' * 50)

# 简单日志类
class TestLogger:
    def info(self, msg): print(f'[INFO] {msg}')
    def warning(self, msg): print(f'[WARNING] {msg}')
    def error(self, msg): print(f'[ERROR] {msg}')
    def debug(self, msg): pass

try:
    from ai_models.autoencoder_model import AutoencoderModel
    
    # 初始化自编码器
    config = {
        'model_path': 'models/autoencoder_model_retrained',
        'threshold': 0.489394,
        'input_dim': 6
    }
    
    autoencoder = AutoencoderModel(config, TestLogger())
    print('✅ 自编码器初始化成功')
    
    # 加载测试数据
    df = pd.read_csv('data/improved_training_data_6d.csv')
    feature_columns = ['avg_signal_strength', 'avg_data_rate', 'avg_latency', 
                      'packet_loss_rate', 'system_load', 'network_stability']
    
    # 测试正常数据
    normal_data = df[df['label'] == 0].sample(n=10, random_state=42)
    normal_correct = 0
    
    print('\n📊 正常数据测试:')
    for i, (_, sample) in enumerate(normal_data.iterrows()):
        features = sample[feature_columns].values
        result = autoencoder.predict(features)
        
        is_anomaly = result['is_anomaly']
        reconstruction_error = result['reconstruction_error']
        
        if not is_anomaly:
            normal_correct += 1
        
        status = '正常' if not is_anomaly else '异常'
        icon = '✅' if not is_anomaly else '❌'
        print(f'  {icon} 样本{i+1}: 重构误差={reconstruction_error:.6f} → {status}')
    
    # 测试异常数据
    anomaly_data = df[df['label'] == 1].sample(n=10, random_state=42)
    anomaly_correct = 0
    
    print('\n📊 异常数据测试:')
    for i, (_, sample) in enumerate(anomaly_data.iterrows()):
        features = sample[feature_columns].values
        result = autoencoder.predict(features)
        
        is_anomaly = result['is_anomaly']
        reconstruction_error = result['reconstruction_error']
        anomaly_type = sample['anomaly_type']
        
        if is_anomaly:
            anomaly_correct += 1
        
        status = '异常' if is_anomaly else '正常'
        icon = '✅' if is_anomaly else '❌'
        print(f'  {icon} 样本{i+1}: {anomaly_type} 重构误差={reconstruction_error:.6f} → {status}')
    
    # 性能统计
    normal_accuracy = normal_correct / 10 * 100
    anomaly_accuracy = anomaly_correct / 10 * 100
    overall_accuracy = (normal_correct + anomaly_correct) / 20 * 100
    
    print(f'\n📈 自编码器性能:')
    print(f'  正常数据准确率: {normal_accuracy:.1f}% ({normal_correct}/10)')
    print(f'  异常数据准确率: {anomaly_accuracy:.1f}% ({anomaly_correct}/10)')
    print(f'  总体准确率: {overall_accuracy:.1f}%')
    print(f'  检测阈值: {config["threshold"]}')
    
    # 性能等级
    if overall_accuracy >= 90:
        grade = '🌟 卓越 (A+)'
    elif overall_accuracy >= 80:
        grade = '⭐ 优秀 (A)'
    elif overall_accuracy >= 70:
        grade = '✅ 良好 (B+)'
    else:
        grade = '⚠️ 需改进'
    
    print(f'  自编码器等级: {grade}')
    print(f'\n✅ 自编码器测试完成')
    
except Exception as e:
    print(f'❌ 测试失败: {e}')
    import traceback
    traceback.print_exc()

print('=' * 50) 