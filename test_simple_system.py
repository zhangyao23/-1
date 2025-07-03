#!/usr/bin/env python3
"""简化的系统测试脚本"""

import sys
import os
import numpy as np

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai_models.autoencoder_model import AutoencoderModel
from ai_models.error_classifier import ErrorClassifier

print("🔧 简化系统测试")
print("=" * 60)

# 1. 初始化AI模型
print("📦 初始化AI模型...")
try:
    # 简化的日志记录器
    class SimpleLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def debug(self, msg): pass
        def error(self, msg): print(f"[ERROR] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
    
    logger = SimpleLogger()
    
    # 加载系统配置
    import json
    with open('config/system_config.json', 'r', encoding='utf-8') as f:
        system_config = json.load(f)
    
    # 初始化自编码器
    autoencoder_config = system_config['ai_models']['autoencoder']
    autoencoder = AutoencoderModel(autoencoder_config, logger)
    print("✅ 自编码器初始化成功")
    
    # 初始化分类器
    classifier_config = system_config['ai_models']['classifier']
    classifier = ErrorClassifier(classifier_config, logger)
    print("✅ 分类器初始化成功")
    
except Exception as e:
    print(f"❌ 模型初始化失败: {e}")
    exit(1)

# 2. 准备测试特征数据（6维）
print("\n🎯 准备测试特征数据...")
try:
    # 正常网络特征（基于训练数据的正常范围）
    normal_features = np.array([
        [8.0, 2.5, 10.0, 0.001, 0.8, 0.95]
    ])
    
    # 异常网络特征（高延迟、高丢包率）
    anomaly_features = np.array([
        [2.0, 1.0, 80.0, 0.15, 0.9, 0.60]
    ])
    
    print(f"正常特征: {normal_features[0]}")
    print(f"异常特征: {anomaly_features[0]}")
    
except Exception as e:
    print(f"❌ 测试数据准备失败: {e}")
    exit(1)

# 3. 测试自编码器异常检测
print("\n🤖 测试自编码器异常检测...")

def test_autoencoder(features, data_type):
    """测试自编码器"""
    print(f"\n--- 测试{data_type}特征 ---")
    
    try:
        # 异常检测
        result = autoencoder.predict(features)
        
        print(f"  重构误差: {result['reconstruction_error']:.6f}")
        print(f"  异常阈值: {result['threshold']:.6f}")
        print(f"  是否异常: {result['is_anomaly']}")
        print(f"  异常分数: {result.get('anomaly_score', 'N/A')}")
        
        return result['is_anomaly']
        
    except Exception as e:
        print(f"❌ {data_type}特征检测失败: {e}")
        return False

# 测试正常特征
normal_is_anomaly = test_autoencoder(normal_features, "正常")

# 测试异常特征
anomaly_is_anomaly = test_autoencoder(anomaly_features, "异常")

# 4. 测试分类器
print("\n🏷️ 测试分类器...")

def test_classifier(features, data_type):
    """测试分类器"""
    print(f"\n--- 分类{data_type}特征 ---")
    
    try:
        # 异常分类
        result = classifier.classify_error(features)
        
        print(f"  预测类别: {result['predicted_class']}")
        print(f"  置信度: {result['confidence']:.4f}")
        if 'class_probabilities' in result:
            probs = list(result['class_probabilities'].values())
            print(f"  所有概率: {[f'{p:.3f}' for p in probs]}")
        else:
            print(f"  概率信息: 不可用")
        
        return result['predicted_class']
        
    except Exception as e:
        print(f"❌ {data_type}特征分类失败: {e}")
        return None

# 只对异常特征进行分类测试
if anomaly_is_anomaly:
    anomaly_class = test_classifier(anomaly_features, "异常")
else:
    print("⚠️ 自编码器未检测到异常，跳过分类测试")
    anomaly_class = None

# 测试正常特征的分类（看看分类器如何处理）
normal_class = test_classifier(normal_features, "正常")

# 5. 批量测试
print("\n📊 批量测试多种特征...")
test_cases = [
    {
        'name': '标准正常',
        'features': np.array([[8.5, 2.8, 8.0, 0.001, 0.6, 0.98]])
    },
    {
        'name': '轻微异常',
        'features': np.array([[7.0, 2.0, 15.0, 0.02, 0.7, 0.85]])
    },
    {
        'name': '严重异常',
        'features': np.array([[4.0, 0.8, 50.0, 0.25, 0.9, 0.45]])
    },
    {
        'name': '边界情况',
        'features': np.array([[6.0, 1.5, 20.0, 0.05, 0.8, 0.70]])
    }
]

batch_results = []
for i, test_case in enumerate(test_cases):
    print(f"\n--- 批量测试 {i+1}/{len(test_cases)}: {test_case['name']} ---")
    
    try:
        # 异常检测
        detection_result = autoencoder.predict(test_case['features'])
        
        result = {
            'name': test_case['name'],
            'reconstruction_error': detection_result['reconstruction_error'],
            'is_anomaly': detection_result['is_anomaly'],
            'anomaly_type': None,
            'confidence': None
        }
        
        print(f"  重构误差: {detection_result['reconstruction_error']:.6f}")
        print(f"  是否异常: {detection_result['is_anomaly']}")
        
        if detection_result['is_anomaly']:
            classification_result = classifier.classify_error(test_case['features'])
            result['anomaly_type'] = classification_result['predicted_class']
            result['confidence'] = classification_result['confidence']
            print(f"  异常类型: {classification_result['predicted_class']}")
            print(f"  置信度: {classification_result['confidence']:.4f}")
        
        batch_results.append(result)
        
    except Exception as e:
        print(f"❌ 批量测试失败: {e}")
        batch_results.append({
            'name': test_case['name'],
            'error': str(e)
        })

# 6. 总结报告
print("\n" + "=" * 60)
print("📋 系统测试总结报告")
print("=" * 60)

print(f"\n🎯 基本功能测试:")
print(f"  自编码器对正常数据: {'✅ 正确' if not normal_is_anomaly else '❌ 误判'}")
print(f"  自编码器对异常数据: {'✅ 正确' if anomaly_is_anomaly else '❌ 漏检'}")
print(f"  分类器功能状态: {'✅ 正常' if anomaly_class is not None else '❌ 异常'}")

print(f"\n📊 批量测试结果:")
success_count = 0
for result in batch_results:
    if 'error' in result:
        print(f"  ❌ {result['name']}: {result['error']}")
    else:
        status = "异常" if result['is_anomaly'] else "正常"
        if result['is_anomaly']:
            print(f"  🔴 {result['name']}: {status} -> {result['anomaly_type']} (置信度: {result['confidence']:.3f})")
        else:
            print(f"  🟢 {result['name']}: {status}")
        success_count += 1

print(f"\n🎯 系统状态评估:")
if success_count == len(batch_results) and anomaly_is_anomaly and not normal_is_anomaly:
    print("✅ 系统完全正常工作")
    print("✅ 自编码器异常检测功能正常")
    print("✅ 分类器异常分类功能正常")
    print("✅ 端到端AI推理管道正常")
else:
    print("⚠️ 系统基本功能正常，但可能需要优化")
    if not anomaly_is_anomaly:
        print("⚠️ 自编码器可能存在漏检问题")
    if normal_is_anomaly:
        print("⚠️ 自编码器可能存在误报问题")
    if anomaly_class is None:
        print("⚠️ 分类器可能存在问题")

print(f"\n✅ 成功测试: {success_count}/{len(batch_results)} 个场景")
print("\n" + "=" * 60)
print("🏁 简化系统测试完成") 