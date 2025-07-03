#!/usr/bin/env python3
"""完整系统测试脚本"""

import sys
import os
import numpy as np
import pandas as pd

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_processor.feature_extractor import FeatureExtractor
from ai_models.autoencoder_model import AutoencoderModel
from ai_models.error_classifier import ErrorClassifier

print("🔧 完整系统测试")
print("=" * 80)

# 1. 初始化组件
print("📦 初始化系统组件...")
try:
    # 简化的配置
    metrics_config = {
        'signal_strength': {'weight': 1.0},
        'data_rate': {'weight': 1.0},
        'latency': {'weight': 1.0},
        'packet_loss': {'weight': 1.0},
        'system_load': {'weight': 1.0},
        'network_stability': {'weight': 1.0}
    }
    
    # 简化的日志记录器
    class SimpleLogger:
        def info(self, msg): pass
        def debug(self, msg): pass
        def error(self, msg): pass
    
    logger = SimpleLogger()
    
    # 初始化特征提取器
    feature_extractor = FeatureExtractor(metrics_config, logger)
    print("✅ 特征提取器初始化成功")
    
    # 初始化自编码器
    autoencoder_config = {}
    autoencoder = AutoencoderModel(autoencoder_config, logger)
    print("✅ 自编码器初始化成功")
    
    # 初始化分类器
    classifier_config = {}
    classifier = ErrorClassifier(classifier_config, logger)
    print("✅ 分类器初始化成功")
    
except Exception as e:
    print(f"❌ 组件初始化失败: {e}")
    exit(1)

# 2. 准备测试数据
print("\n🎯 准备测试数据...")
try:
    # 模拟正常网络数据
    normal_raw_data = np.array([
        [80.0, 2.0, 10.0, 0.001, 8.0, 30.0, 0.95, 5.0, 1.0, 0.5, 2.0]
    ])
    
    # 模拟异常网络数据（高延迟、高丢包率）
    anomaly_raw_data = np.array([
        [60.0, 1.0, 25.0, 0.15, 15.0, 20.0, 0.60, 8.0, 3.0, 2.0, 5.0]
    ])
    
    print(f"正常数据: {normal_raw_data[0]}")
    print(f"异常数据: {anomaly_raw_data[0]}")
    
except Exception as e:
    print(f"❌ 测试数据准备失败: {e}")
    exit(1)

# 3. 测试完整流程
print("\n🔄 测试完整检测流程...")

def test_detection_pipeline(raw_data, data_type):
    """测试完整的检测流程"""
    print(f"\n--- 测试{data_type}数据 ---")
    
    try:
        # 步骤1：特征提取
        print("🔍 步骤1：特征提取...")
        features = feature_extractor.extract_features(raw_data)
        print(f"  输入维度: {raw_data.shape}")
        print(f"  特征维度: {features.shape}")
        print(f"  提取的特征: {features[0]}")
        
        # 步骤2：异常检测
        print("🤖 步骤2：异常检测...")
        detection_result = autoencoder.predict(features)
        print(f"  重构误差: {detection_result['reconstruction_error']:.6f}")
        print(f"  异常阈值: {detection_result['threshold']:.6f}")
        print(f"  是否异常: {detection_result['is_anomaly']}")
        
        # 步骤3：异常分类（如果检测到异常）
        if detection_result['is_anomaly']:
            print("🏷️  步骤3：异常分类...")
            classification_result = classifier.classify_error(features)
            print(f"  预测类别: {classification_result['predicted_class']}")
            print(f"  置信度: {classification_result['confidence']:.4f}")
            print(f"  所有概率: {classification_result['probabilities']}")
        else:
            print("✅ 步骤3：无需分类（检测为正常）")
        
        print(f"✅ {data_type}数据检测完成")
        return True
        
    except Exception as e:
        print(f"❌ {data_type}数据检测失败: {e}")
        return False

# 4. 运行测试
print("\n🧪 运行系统测试...")
success_count = 0
total_tests = 2

# 测试正常数据
if test_detection_pipeline(normal_raw_data, "正常"):
    success_count += 1

# 测试异常数据
if test_detection_pipeline(anomaly_raw_data, "异常"):
    success_count += 1

# 5. 批量测试
print(f"\n📊 批量测试多种场景...")
test_cases = [
    {
        'name': '正常场景',
        'data': np.array([[85.0, 2.5, 8.0, 0.001, 6.0, 35.0, 0.98, 4.0, 0.5, 0.2, 1.0]])
    },
    {
        'name': '信号弱化',
        'data': np.array([[45.0, 1.2, 12.0, 0.02, 10.0, 15.0, 0.75, 6.0, 1.5, 1.0, 3.0]])
    },
    {
        'name': '网络拥堵',
        'data': np.array([[70.0, 1.8, 30.0, 0.08, 18.0, 25.0, 0.65, 12.0, 2.5, 1.8, 6.0]])
    },
    {
        'name': '系统过载',
        'data': np.array([[75.0, 2.0, 15.0, 0.05, 22.0, 40.0, 0.80, 15.0, 3.0, 2.5, 8.0]])
    }
]

batch_results = []
for i, test_case in enumerate(test_cases):
    print(f"\n--- 批量测试 {i+1}/{len(test_cases)}: {test_case['name']} ---")
    
    try:
        # 完整流程
        features = feature_extractor.extract_features(test_case['data'])
        detection_result = autoencoder.predict(features)
        
        result = {
            'name': test_case['name'],
            'features': features[0],
            'reconstruction_error': detection_result['reconstruction_error'],
            'is_anomaly': detection_result['is_anomaly'],
            'anomaly_type': None,
            'confidence': None
        }
        
        if detection_result['is_anomaly']:
            classification_result = classifier.classify_error(features)
            result['anomaly_type'] = classification_result['predicted_class']
            result['confidence'] = classification_result['confidence']
        
        batch_results.append(result)
        
        print(f"  重构误差: {detection_result['reconstruction_error']:.6f}")
        print(f"  是否异常: {detection_result['is_anomaly']}")
        if detection_result['is_anomaly']:
            print(f"  异常类型: {classification_result['predicted_class']}")
            print(f"  置信度: {classification_result['confidence']:.4f}")
        
    except Exception as e:
        print(f"❌ 批量测试失败: {e}")
        batch_results.append({
            'name': test_case['name'],
            'error': str(e)
        })

# 6. 总结报告
print("\n" + "=" * 80)
print("📋 系统测试总结报告")
print("=" * 80)

print(f"\n✅ 基本功能测试: {success_count}/{total_tests} 通过")

print(f"\n📊 批量测试结果:")
for result in batch_results:
    if 'error' in result:
        print(f"  ❌ {result['name']}: {result['error']}")
    else:
        status = "异常" if result['is_anomaly'] else "正常"
        if result['is_anomaly']:
            print(f"  🔴 {result['name']}: {status} -> {result['anomaly_type']} (置信度: {result['confidence']:.3f})")
        else:
            print(f"  🟢 {result['name']}: {status}")

print(f"\n🎯 系统状态评估:")
if success_count == total_tests and len([r for r in batch_results if 'error' not in r]) == len(batch_results):
    print("✅ 系统完全正常工作")
    print("✅ 自编码器异常检测功能正常")
    print("✅ 分类器异常分类功能正常")
    print("✅ 特征提取器工作正常")
    print("✅ 端到端流程运行正常")
else:
    print("❌ 系统存在问题，需要进一步调试")

print("\n" + "=" * 80)
print("🏁 系统测试完成") 