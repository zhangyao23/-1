#!/usr/bin/env python3
"""
改进模型测试脚本
验证新训练的6维模型的检测效果和不同阈值策略
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def load_improved_model():
    """加载改进的模型组件"""
    print("🔄 加载改进的模型组件...")
    
    # 加载自编码器
    autoencoder_path = 'models/autoencoder_model_improved/autoencoder_model.keras'
    autoencoder = tf.keras.models.load_model(autoencoder_path)
    print(f"✅ 自编码器已加载: {autoencoder_path}")
    
    # 加载标准化器
    scaler_path = 'models/autoencoder_model_improved/autoencoder_scaler.pkl'
    scaler = joblib.load(scaler_path)
    print(f"✅ 标准化器已加载: {scaler_path}")
    
    # 加载阈值配置
    threshold_path = 'models/autoencoder_model_improved/threshold_config.json'
    with open(threshold_path, 'r') as f:
        threshold_config = json.load(f)
    print(f"✅ 阈值配置已加载: {threshold_path}")
    
    # 加载分类器
    classifier_path = 'models/rf_classifier_improved.pkl'
    classifier = joblib.load(classifier_path)
    print(f"✅ 分类器已加载: {classifier_path}")
    
    return autoencoder, scaler, threshold_config, classifier

def generate_test_samples():
    """生成测试样本"""
    print("📊 生成测试样本...")
    
    test_samples = []
    
    # 正常样本
    for i in range(5):
        sample = {
            'name': f'正常样本{i+1}',
            'type': 'normal',
            'data': np.array([
                np.random.normal(7.0, 1.0),    # avg_signal_strength (正常范围)
                np.random.normal(2.5, 0.3),    # avg_data_rate
                np.random.normal(15.0, 3.0),   # avg_latency
                np.random.normal(0.02, 0.01),  # packet_loss_rate
                np.random.normal(0.3, 0.1),    # system_load
                np.random.normal(0.85, 0.05)   # network_stability
            ])
        }
        test_samples.append(sample)
    
    # 信号降级异常
    for i in range(3):
        sample = {
            'name': f'信号降级异常{i+1}',
            'type': 'signal_degradation',
            'data': np.array([
                np.random.normal(3.0, 0.5),    # 信号弱
                np.random.normal(1.5, 0.3),    # 数据率低
                np.random.normal(25.0, 5.0),   # 延迟高
                np.random.normal(0.08, 0.02),  # 丢包多
                np.random.normal(0.3, 0.1),    # 系统负载正常
                np.random.normal(0.6, 0.1)     # 稳定性差
            ])
        }
        test_samples.append(sample)
    
    # 网络拥堵异常
    for i in range(3):
        sample = {
            'name': f'网络拥堵异常{i+1}',
            'type': 'network_congestion',
            'data': np.array([
                np.random.normal(6.5, 0.5),    # 信号正常
                np.random.normal(1.2, 0.2),    # 数据率极低
                np.random.normal(40.0, 8.0),   # 延迟极高
                np.random.normal(0.12, 0.03),  # 丢包严重
                np.random.normal(0.7, 0.1),    # 负载高
                np.random.normal(0.4, 0.1)     # 很不稳定
            ])
        }
        test_samples.append(sample)
    
    # 资源过载异常
    for i in range(3):
        sample = {
            'name': f'资源过载异常{i+1}',
            'type': 'resource_overload',
            'data': np.array([
                np.random.normal(6.5, 0.5),    # 信号正常
                np.random.normal(1.8, 0.3),    # 数据率偏低
                np.random.normal(35.0, 6.0),   # 延迟高
                np.random.normal(0.06, 0.02),  # 丢包偏多
                np.random.normal(0.9, 0.05),   # 极高负载
                np.random.normal(0.65, 0.1)    # 稳定性差
            ])
        }
        test_samples.append(sample)
    
    # 确保数值在合理范围内
    for sample in test_samples:
        data = sample['data']
        data[0] = np.clip(data[0], 0.5, 10.0)   # signal_strength
        data[1] = np.clip(data[1], 0.1, 5.0)    # data_rate
        data[2] = np.clip(data[2], 1.0, 100.0)  # latency
        data[3] = np.clip(data[3], 0.0, 0.5)    # packet_loss
        data[4] = np.clip(data[4], 0.0, 1.0)    # system_load
        data[5] = np.clip(data[5], 0.0, 1.0)    # network_stability
        sample['data'] = data
    
    print(f"✅ 生成了 {len(test_samples)} 个测试样本")
    return test_samples

def test_anomaly_detection(autoencoder, scaler, threshold_config, classifier, test_samples):
    """测试异常检测效果"""
    print("\n🧪 开始异常检测测试...")
    
    feature_names = [
        'avg_signal_strength', 'avg_data_rate', 'avg_latency',
        'packet_loss_rate', 'system_load', 'network_stability'
    ]
    
    anomaly_types = [
        'signal_degradation', 'network_congestion', 'connection_timeout',
        'packet_corruption', 'resource_overload', 'mixed_anomaly'
    ]
    
    results = []
    
    print("=" * 100)
    print(f"{'样本名称':<20} {'预期类型':<15} {'重构误差':<12} {'是否异常':<8} {'异常类型':<15} {'置信度':<8}")
    print("=" * 100)
    
    for sample in test_samples:
        # 标准化数据
        data_scaled = scaler.transform(sample['data'].reshape(1, -1))
        
        # 计算重构误差
        reconstruction = autoencoder.predict(data_scaled, verbose=0)
        mse = np.mean((data_scaled - reconstruction) ** 2)
        
        # 测试不同阈值策略
        threshold_results = {}
        for threshold_name, threshold_value in threshold_config['all_thresholds'].items():
            is_anomaly = mse > threshold_value
            threshold_results[threshold_name] = is_anomaly
        
        # 使用默认阈值（95%）
        default_threshold = threshold_config['selected_threshold']
        is_anomaly = mse > default_threshold
        
        # 如果检测为异常，进行分类
        predicted_class = 'normal'
        confidence = 0.0
        
        if is_anomaly:
            try:
                class_probs = classifier.predict_proba(data_scaled)[0]
                predicted_class_idx = np.argmax(class_probs)
                predicted_class = anomaly_types[predicted_class_idx]
                confidence = class_probs[predicted_class_idx]
            except:
                predicted_class = 'unknown'
                confidence = 0.0
        
        # 记录结果
        result = {
            'name': sample['name'],
            'expected_type': sample['type'],
            'mse': mse,
            'is_anomaly': is_anomaly,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'threshold_results': threshold_results,
            'data': sample['data']
        }
        results.append(result)
        
        # 显示结果
        anomaly_status = "🚨异常" if is_anomaly else "✅正常"
        print(f"{sample['name']:<20} {sample['type']:<15} {mse:<12.6f} {anomaly_status:<8} {predicted_class:<15} {confidence:<8.3f}")
    
    print("=" * 100)
    return results

def analyze_threshold_strategies(results, threshold_config):
    """分析不同阈值策略的效果"""
    print("\n📊 阈值策略分析...")
    
    threshold_analysis = {}
    
    for threshold_name, threshold_value in threshold_config['all_thresholds'].items():
        true_positives = 0  # 正确识别的异常
        false_positives = 0  # 误报（正常识别为异常）
        true_negatives = 0   # 正确识别的正常
        false_negatives = 0  # 漏报（异常识别为正常）
        
        for result in results:
            is_anomaly_predicted = result['threshold_results'][threshold_name]
            is_anomaly_actual = result['expected_type'] != 'normal'
            
            if is_anomaly_actual and is_anomaly_predicted:
                true_positives += 1
            elif not is_anomaly_actual and is_anomaly_predicted:
                false_positives += 1
            elif not is_anomaly_actual and not is_anomaly_predicted:
                true_negatives += 1
            elif is_anomaly_actual and not is_anomaly_predicted:
                false_negatives += 1
        
        # 计算指标
        total_anomalies = true_positives + false_negatives
        total_normal = true_negatives + false_positives
        
        sensitivity = true_positives / total_anomalies if total_anomalies > 0 else 0
        specificity = true_negatives / total_normal if total_normal > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(results)
        
        threshold_analysis[threshold_name] = {
            'threshold_value': threshold_value,
            'sensitivity': sensitivity,      # 检出率
            'specificity': specificity,      # 特异性
            'precision': precision,          # 精确率
            'accuracy': accuracy,            # 准确率
            'false_positive_rate': 1 - specificity,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    
    # 显示分析结果
    print(f"\n{'策略名称':<15} {'阈值':<12} {'检出率':<8} {'误报率':<8} {'精确率':<8} {'准确率':<8}")
    print("-" * 70)
    
    for name, analysis in threshold_analysis.items():
        print(f"{name:<15} {analysis['threshold_value']:<12.6f} "
              f"{analysis['sensitivity']:<8.3f} {analysis['false_positive_rate']:<8.3f} "
              f"{analysis['precision']:<8.3f} {analysis['accuracy']:<8.3f}")
    
    return threshold_analysis

def test_classification_accuracy(results):
    """测试分类准确性"""
    print("\n🎯 异常分类准确性分析...")
    
    # 统计分类结果
    classification_results = {}
    total_anomalies = 0
    correct_classifications = 0
    
    for result in results:
        if result['expected_type'] != 'normal' and result['is_anomaly']:
            total_anomalies += 1
            expected = result['expected_type']
            predicted = result['predicted_class']
            
            if expected not in classification_results:
                classification_results[expected] = {'total': 0, 'correct': 0, 'predictions': {}}
            
            classification_results[expected]['total'] += 1
            
            if predicted not in classification_results[expected]['predictions']:
                classification_results[expected]['predictions'][predicted] = 0
            classification_results[expected]['predictions'][predicted] += 1
            
            if expected == predicted:
                classification_results[expected]['correct'] += 1
                correct_classifications += 1
    
    # 显示分类结果
    print(f"\n{'预期类型':<20} {'总数':<6} {'正确':<6} {'准确率':<8} {'主要预测'}")
    print("-" * 60)
    
    for anomaly_type, stats in classification_results.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        main_prediction = max(stats['predictions'].items(), key=lambda x: x[1])[0] if stats['predictions'] else 'N/A'
        print(f"{anomaly_type:<20} {stats['total']:<6} {stats['correct']:<6} {accuracy:<8.3f} {main_prediction}")
    
    overall_accuracy = correct_classifications / total_anomalies if total_anomalies > 0 else 0
    print(f"\n📈 整体分类准确率: {overall_accuracy:.3f} ({correct_classifications}/{total_anomalies})")
    
    return classification_results

def display_feature_analysis(results):
    """显示特征分析"""
    print("\n📈 特征值分析...")
    
    feature_names = [
        'avg_signal_strength', 'avg_data_rate', 'avg_latency',
        'packet_loss_rate', 'system_load', 'network_stability'
    ]
    
    # 按类型分组
    normal_features = []
    anomaly_features = []
    
    for result in results:
        if result['expected_type'] == 'normal':
            normal_features.append(result['data'])
        else:
            anomaly_features.append(result['data'])
    
    if normal_features and anomaly_features:
        normal_features = np.array(normal_features)
        anomaly_features = np.array(anomaly_features)
        
        print(f"\n{'特征名称':<20} {'正常均值':<10} {'异常均值':<10} {'差异':<8}")
        print("-" * 50)
        
        for i, feature_name in enumerate(feature_names):
            normal_mean = np.mean(normal_features[:, i])
            anomaly_mean = np.mean(anomaly_features[:, i])
            difference = abs(anomaly_mean - normal_mean)
            
            print(f"{feature_name:<20} {normal_mean:<10.3f} {anomaly_mean:<10.3f} {difference:<8.3f}")

def main():
    """主测试函数"""
    print("🚀 改进模型效果测试")
    print("=" * 50)
    
    try:
        # 加载模型
        autoencoder, scaler, threshold_config, classifier = load_improved_model()
        
        # 生成测试样本
        test_samples = generate_test_samples()
        
        # 执行异常检测测试
        results = test_anomaly_detection(autoencoder, scaler, threshold_config, classifier, test_samples)
        
        # 分析阈值策略
        threshold_analysis = analyze_threshold_strategies(results, threshold_config)
        
        # 测试分类准确性
        classification_results = test_classification_accuracy(results)
        
        # 特征分析
        display_feature_analysis(results)
        
        print("\n🎉 测试完成！")
        print("📊 测试总结:")
        print(f"   - 测试样本总数: {len(test_samples)}")
        print(f"   - 正常样本: {len([r for r in results if r['expected_type'] == 'normal'])}")
        print(f"   - 异常样本: {len([r for r in results if r['expected_type'] != 'normal'])}")
        print(f"   - 可用阈值策略: {len(threshold_config['all_thresholds'])}")
        
        # 推荐最佳阈值策略
        best_strategy = None
        best_score = 0
        
        for name, analysis in threshold_analysis.items():
            # 平衡检出率和误报率的得分
            score = analysis['sensitivity'] * 0.7 + (1 - analysis['false_positive_rate']) * 0.3
            if score > best_score:
                best_score = score
                best_strategy = name
        
        if best_strategy:
            print(f"\n💡 推荐阈值策略: {best_strategy}")
            print(f"   检出率: {threshold_analysis[best_strategy]['sensitivity']:.3f}")
            print(f"   误报率: {threshold_analysis[best_strategy]['false_positive_rate']:.3f}")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 