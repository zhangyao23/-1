#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分类器偏差诊断脚本
分析为什么分类器只能识别一种异常类型
"""

import numpy as np
import pandas as pd
import sys
import os
import joblib
from collections import Counter

# 添加源代码路径
sys.path.append('src')
sys.path.append('.')

print('🔍 分类器偏差问题诊断')
print('=' * 60)

# ========================================
# 第1步：分析训练数据分布
# ========================================
print('\n📊 第1步：分析训练数据分布')
print('-' * 40)

try:
    # 加载训练数据
    df = pd.read_csv('data/improved_training_data_6d.csv')
    print(f'✅ 加载训练数据: {df.shape}')
    
    # 分析异常类型分布
    anomaly_data = df[df['label'] == 1]
    print(f'📈 异常样本总数: {len(anomaly_data)}')
    
    if 'anomaly_type' in anomaly_data.columns:
        type_distribution = anomaly_data['anomaly_type'].value_counts()
        print(f'🏷️ 异常类型分布:')
        for anomaly_type, count in type_distribution.items():
            percentage = count / len(anomaly_data) * 100
            print(f'  {anomaly_type}: {count} 样本 ({percentage:.1f}%)')
            
        # 检查数据平衡性
        max_count = type_distribution.max()
        min_count = type_distribution.min()
        imbalance_ratio = max_count / min_count
        print(f'📊 数据不平衡比例: {imbalance_ratio:.2f}:1 (最多:最少)')
        
        if imbalance_ratio > 3:
            print('⚠️ 数据严重不平衡！')
        elif imbalance_ratio > 1.5:
            print('⚠️ 数据轻度不平衡')
        else:
            print('✅ 数据分布相对平衡')
    else:
        print('❌ 训练数据中没有anomaly_type列')
        
except Exception as e:
    print(f'❌ 加载训练数据失败: {e}')

# ========================================
# 第2步：分析模型文件内容
# ========================================
print('\n🔧 第2步：分析模型文件内容')
print('-' * 40)

try:
    model_data = joblib.load('models/rf_classifier_improved.pkl')
    print(f'📦 模型文件类型: {type(model_data)}')
    
    if isinstance(model_data, dict):
        print('📋 模型文件内容:')
        for key, value in model_data.items():
            print(f'  {key}: {type(value)}')
            
            if key == 'classes':
                print(f'    支持的类别: {value}')
            elif key == 'training_info':
                print(f'    训练信息: {value}')
            elif key == 'model' and hasattr(value, 'classes_'):
                print(f'    sklearn模型类别: {value.classes_}')
                print(f'    sklearn特征数: {getattr(value, "n_features_in_", "未知")}')
                
                # 分析特征重要性
                if hasattr(value, 'feature_importances_'):
                    feature_importance = value.feature_importances_
                    feature_names = ['avg_signal_strength', 'avg_data_rate', 'avg_latency', 
                                   'packet_loss_rate', 'system_load', 'network_stability']
                    print(f'    特征重要性:')
                    for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
                        print(f'      {name}: {importance:.4f}')
                        
except Exception as e:
    print(f'❌ 分析模型文件失败: {e}')

# ========================================
# 第3步：分析测试数据分布
# ========================================
print('\n📈 第3步：分析测试数据分布')
print('-' * 40)

try:
    # 加载测试数据
    df_test = pd.read_csv('data/improved_training_data_6d.csv')
    anomaly_test = df_test[df_test['label'] == 1]
    
    # 随机选择不同类型的异常样本进行测试
    if 'anomaly_type' in anomaly_test.columns:
        unique_types = anomaly_test['anomaly_type'].unique()
        print(f'🎯 可用的异常类型: {unique_types}')
        
        # 为每种类型选择代表性样本
        test_samples = {}
        for anomaly_type in unique_types:
            type_samples = anomaly_test[anomaly_test['anomaly_type'] == anomaly_type]
            if len(type_samples) > 0:
                # 选择第一个样本作为代表
                sample = type_samples.iloc[0]
                features = sample.drop(['label', 'anomaly_type']).values
                test_samples[anomaly_type] = features
                print(f'  {anomaly_type}: 样本范围 {features.min():.3f} - {features.max():.3f}')
    else:
        print('❌ 测试数据中没有anomaly_type列')
        test_samples = {}
        
except Exception as e:
    print(f'❌ 分析测试数据失败: {e}')
    test_samples = {}

# ========================================
# 第4步：测试分类器对不同类型的预测
# ========================================
print('\n🧪 第4步：测试分类器对不同类型的预测')
print('-' * 40)

try:
    # 简单日志类
    class SimpleLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
        def debug(self, msg): print(f"[DEBUG] {msg}")

    from ai_models.error_classifier import ErrorClassifier
    
    # 初始化分类器
    classifier_config = {
        'model_path': 'models/rf_classifier_improved.pkl',
        'classes': ['connection_timeout', 'mixed_anomaly', 'network_congestion', 
                   'packet_corruption', 'resource_overload', 'signal_degradation'],
        'confidence_threshold': 0.7
    }
    
    classifier = ErrorClassifier(classifier_config, SimpleLogger())
    
    if test_samples:
        print('🔍 测试不同类型异常的分类结果:')
        prediction_counts = Counter()
        
        for true_type, features in test_samples.items():
            result = classifier.classify_error(features)
            predicted_type = result['predicted_class']
            confidence = result['confidence']
            
            prediction_counts[predicted_type] += 1
            
            print(f'  真实类型: {true_type:20} → 预测类型: {predicted_type:20} (置信度: {confidence:.3f})')
            
        print(f'\n📊 预测结果统计:')
        for pred_type, count in prediction_counts.items():
            percentage = count / len(test_samples) * 100
            print(f'  {pred_type}: {count}/{len(test_samples)} ({percentage:.1f}%)')
            
        # 分析预测偏差
        if len(prediction_counts) == 1:
            print(f'❌ 严重问题：所有样本都被预测为 {list(prediction_counts.keys())[0]}')
        elif len(prediction_counts) < len(unique_types) / 2:
            print(f'⚠️ 偏差问题：只预测了 {len(prediction_counts)} 种类型，实际有 {len(unique_types)} 种')
        else:
            print(f'✅ 预测多样性正常：预测了 {len(prediction_counts)} 种类型')
            
    else:
        print('❌ 无法获取测试样本')
        
except Exception as e:
    print(f'❌ 分类器测试失败: {e}')
    import traceback
    traceback.print_exc()

# ========================================
# 第5步：深度分析分类器决策过程
# ========================================
print('\n🧠 第5步：深度分析分类器决策过程')
print('-' * 40)

try:
    if test_samples and len(test_samples) > 0:
        # 选择一个样本进行详细分析
        sample_type, sample_features = list(test_samples.items())[0]
        print(f'🔍 详细分析样本: {sample_type}')
        print(f'📊 特征值: {sample_features}')
        
        # 获取所有类别的概率
        features_reshaped = sample_features.reshape(1, -1)
        probabilities = classifier.classifier.predict_proba(features_reshaped)[0]
        
        print(f'🎯 各类别预测概率:')
        class_names = classifier.label_encoder.classes_
        prob_pairs = list(zip(class_names, probabilities))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for class_name, prob in prob_pairs:
            stars = '★' * int(prob * 10)
            print(f'  {class_name:20}: {prob:.4f} {stars}')
            
        # 检查是否存在明显的概率偏差
        max_prob = max(probabilities)
        second_max_prob = sorted(probabilities, reverse=True)[1]
        prob_gap = max_prob - second_max_prob
        
        print(f'\n📈 概率分析:')
        print(f'  最高概率: {max_prob:.4f}')
        print(f'  第二高概率: {second_max_prob:.4f}')
        print(f'  概率差距: {prob_gap:.4f}')
        
        if prob_gap > 0.8:
            print('❌ 极度偏向单一类别！')
        elif prob_gap > 0.5:
            print('⚠️ 偏向单一类别')
        else:
            print('✅ 概率分布相对合理')
            
except Exception as e:
    print(f'❌ 决策分析失败: {e}')

# ========================================
# 诊断总结
# ========================================
print('\n' + '=' * 60)
print('🏁 诊断总结')
print('=' * 60)

print('''
可能的问题原因：

1️⃣ 训练数据不平衡
   - 某种异常类型样本过多
   - 模型学习偏向多数类

2️⃣ 特征空间重叠
   - 不同异常类型的特征相似
   - 分类器难以区分

3️⃣ 模型过拟合
   - 训练时过度拟合某个类别
   - 泛化能力不足

4️⃣ 特征工程问题
   - 特征对某些异常类型不敏感
   - 需要更多区分性特征

5️⃣ 阈值设置问题
   - 置信度阈值过高/过低
   - 影响分类决策
''')

print('🔬 诊断完成！根据以上结果分析具体问题...') 