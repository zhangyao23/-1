#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI网络异常检测系统 - 性能测试脚本
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# 添加源代码路径
sys.path.append('src')
sys.path.append('.')

# 修正导入路径
try:
    from ai_models.autoencoder_model import AutoencoderModel
    from ai_models.error_classifier import ErrorClassifier
    from logger.system_logger import Logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试其他导入方式...")
    # 直接创建简单日志类
    class Logger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
        def debug(self, msg): pass

print('🚀 AI网络异常检测系统 - 性能测试')
print('=' * 60)

# 初始化日志
logger = Logger()

# 加载测试数据
print('📊 加载测试数据...')
try:
    df = pd.read_csv('data/improved_training_data_6d.csv')
    normal_data = df[df['label'] == 0].drop(['label', 'anomaly_type'], axis=1)
    anomaly_data = df[df['label'] == 1].drop(['label', 'anomaly_type'], axis=1)
    
    # 随机采样
    normal_samples = normal_data.sample(n=min(50, len(normal_data))).values
    anomaly_samples = anomaly_data.sample(n=min(50, len(anomaly_data))).values
    
    print(f'✅ 加载数据: 正常{len(normal_samples)}条, 异常{len(anomaly_samples)}条')
    
    # 显示异常类型分布
    anomaly_types = df[df['label'] == 1]['anomaly_type'].value_counts()
    print('📊 异常类型分布:')
    for atype, count in anomaly_types.items():
        print(f'  {atype}: {count}')
    
except Exception as e:
    print(f'❌ 加载数据失败: {e}')
    print('🔧 使用模拟数据进行测试...')
    
    # 创建模拟数据
    np.random.seed(42)
    normal_samples = np.random.normal(
        loc=[8.0, 2.5, 10.0, 0.001, 0.8, 0.95],
        scale=[1.0, 0.5, 2.0, 0.005, 0.2, 0.1],
        size=(50, 6)
    )
    
    anomaly_samples = np.random.normal(
        loc=[2.0, 1.0, 80.0, 0.15, 0.9, 0.6],
        scale=[1.0, 0.5, 20.0, 0.05, 0.1, 0.2],
        size=(50, 6)
    )
    
    print(f'✅ 生成模拟数据: 正常{len(normal_samples)}条, 异常{len(anomaly_samples)}条')

# 初始化模型
print('🤖 初始化AI模型...')
try:
    # 自编码器配置
    autoencoder_config = {
        'input_features': 6,
        'encoding_dim': 3,
        'threshold': 0.489394,
        'model_path': 'models/autoencoder_model_retrained'
    }
    
    autoencoder = AutoencoderModel(
        config=autoencoder_config,
        logger=logger
    )
    print('✅ 自编码器初始化成功')
    
    # 分类器配置
    classifier_config = {
        'model_path': 'models/rf_classifier_improved.pkl',
        'classes': ['connection_timeout', 'mixed_anomaly', 'network_congestion', 
                   'packet_corruption', 'resource_overload', 'signal_degradation'],
        'confidence_threshold': 0.7
    }
    
    classifier = ErrorClassifier(
        config=classifier_config,
        logger=logger
    )
    print('✅ 分类器初始化成功')
    
except Exception as e:
    print(f'❌ 模型初始化失败: {e}')
    exit(1)

# 测试自编码器性能
print('\n🎯 测试自编码器性能...')
print('-' * 40)

# 测试正常数据
print('📈 测试正常数据...')
normal_errors = []
normal_predictions = []

for i, sample in enumerate(normal_samples):
    try:
        result = autoencoder.predict(sample)
        is_anomaly = result['is_anomaly']
        error = result['reconstruction_error']
        normal_errors.append(error)
        normal_predictions.append(is_anomaly)
        
        if i < 5:
            print(f'  样本 {i+1}: 重构误差={error:.6f}, 异常={is_anomaly}')
            
    except Exception as e:
        print(f'  样本 {i+1} 测试失败: {e}')

# 测试异常数据
print('📉 测试异常数据...')
anomaly_errors = []
anomaly_predictions = []

for i, sample in enumerate(anomaly_samples):
    try:
        result = autoencoder.predict(sample)
        is_anomaly = result['is_anomaly']
        error = result['reconstruction_error']
        anomaly_errors.append(error)
        anomaly_predictions.append(is_anomaly)
        
        if i < 5:
            print(f'  样本 {i+1}: 重构误差={error:.6f}, 异常={is_anomaly}')
            
    except Exception as e:
        print(f'  样本 {i+1} 测试失败: {e}')

# 计算性能指标
if normal_errors and anomaly_errors:
    normal_accuracy = sum(not pred for pred in normal_predictions) / len(normal_predictions)
    anomaly_accuracy = sum(anomaly_predictions) / len(anomaly_predictions)
    overall_accuracy = (normal_accuracy * len(normal_predictions) + 
                      anomaly_accuracy * len(anomaly_predictions)) / (len(normal_predictions) + len(anomaly_predictions))
    
    print(f'\n📊 自编码器性能指标:')
    print(f'  异常检测阈值: {autoencoder.threshold:.6f}')
    print(f'  正常数据准确率: {normal_accuracy:.3f}')
    print(f'  异常数据准确率: {anomaly_accuracy:.3f}')
    print(f'  总体准确率: {overall_accuracy:.3f}')
    print(f'  正常数据重构误差: {np.mean(normal_errors):.6f} ± {np.std(normal_errors):.6f}')
    print(f'  异常数据重构误差: {np.mean(anomaly_errors):.6f} ± {np.std(anomaly_errors):.6f}')

# 测试分类器性能
print('\n🏷️ 测试分类器性能...')
print('-' * 40)

predictions = []
confidences = []

for i, sample in enumerate(anomaly_samples):
    try:
        result = classifier.classify_error(sample)
        pred_class = result['predicted_class']
        confidence = result['confidence']
        predictions.append(pred_class)
        confidences.append(confidence)
        
        if i < 10:
            print(f'  样本 {i+1}: 类别={pred_class}, 置信度={confidence:.3f}')
            
    except Exception as e:
        print(f'  样本 {i+1} 分类失败: {e}')

# 计算分类性能
if predictions:
    known_predictions = [p for p in predictions if p != 'unknown']
    unknown_rate = predictions.count('unknown') / len(predictions)
    
    valid_confidences = [c for c in confidences if c > 0]
    avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
    
    # 统计各类别分布
    class_counts = {}
    for pred in predictions:
        class_counts[pred] = class_counts.get(pred, 0) + 1
    
    print(f'\n📊 分类器性能指标:')
    print(f'  测试样本数: {len(predictions)}')
    print(f'  有效预测数: {len(known_predictions)}')
    print(f'  未知预测率: {unknown_rate:.3f}')
    print(f'  平均置信度: {avg_confidence:.3f}')
    print(f'  类别分布: {class_counts}')

# 测试推理速度
print('\n⚡ 测试推理速度...')
print('-' * 40)

test_sample = normal_samples[0]

# 自编码器速度测试
print('🔥 测试自编码器推理速度...')
ae_times = []
for i in range(100):
    start_time = time.time()
    try:
        autoencoder.predict(test_sample)
        ae_times.append(time.time() - start_time)
    except Exception as e:
        if i < 5:
            print(f'  推理失败 {i+1}: {e}')

# 分类器速度测试
print('🔥 测试分类器推理速度...')
clf_times = []
for i in range(100):
    start_time = time.time()
    try:
        classifier.classify_error(test_sample)
        clf_times.append(time.time() - start_time)
    except Exception as e:
        if i < 5:
            print(f'  分类失败 {i+1}: {e}')

if ae_times and clf_times:
    print(f'\n📊 推理速度基准:')
    print(f'  自编码器平均推理时间: {np.mean(ae_times)*1000:.2f}ms ± {np.std(ae_times)*1000:.2f}ms')
    print(f'  分类器平均推理时间: {np.mean(clf_times)*1000:.2f}ms ± {np.std(clf_times)*1000:.2f}ms')
    print(f'  端到端平均推理时间: {(np.mean(ae_times) + np.mean(clf_times))*1000:.2f}ms')
    print(f'  系统吞吐量: {1.0 / (np.mean(ae_times) + np.mean(clf_times)):.1f} 样本/秒')

# 生成性能评估
print('\n🏆 性能评估总结:')
print('=' * 40)

if normal_errors and anomaly_errors and predictions:
    status = "优秀"
    if overall_accuracy < 0.8:
        status = "需要改进"
    elif unknown_rate > 0.5:
        status = "良好"
    
    print(f'📈 自编码器总体准确率: {overall_accuracy:.3f}')
    print(f'🏷️ 分类器未知预测率: {unknown_rate:.3f}')
    print(f'⚡ 系统处理速度: {1.0 / (np.mean(ae_times) + np.mean(clf_times)):.1f} 样本/秒')
    print(f'🎯 系统健康状态: {status}')
    
    # 性能建议
    print('\n💡 性能优化建议:')
    if overall_accuracy < 0.9:
        print('  - 考虑调整自编码器异常检测阈值')
    if unknown_rate > 0.3:
        print('  - 分类器可能需要更多训练数据')
    if len(ae_times) > 0 and np.mean(ae_times) > 0.1:
        print('  - 考虑模型优化以提高推理速度')

print('\n✅ 全面性能测试完成！') 