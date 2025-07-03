#!/usr/bin/env python3
"""分类器调试脚本"""

import pickle
import numpy as np
import pandas as pd

print("🔍 分类器调试分析")
print("=" * 50)

# 1. 加载分类器
model_data = None
try:
    with open('models/rf_classifier_improved.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    print("✅ 分类器加载成功")
    print(f"数据类型: {type(model_data)}")
    
    if isinstance(model_data, dict):
        print("分类器内容:")
        for key, value in model_data.items():
            print(f"  {key}: {type(value)}")
            if hasattr(value, 'classes_'):
                print(f"    类别: {value.classes_}")
    else:
        print(f"模型对象: {model_data}")
        if hasattr(model_data, 'classes_'):
            print(f"类别: {model_data.classes_}")
            
except Exception as e:
    print(f"❌ 加载分类器失败: {e}")
    model_data = None

print()

# 2. 检查训练数据的类别分布
try:
    df = pd.read_csv('data/improved_training_data_6d.csv')
    print("📊 训练数据异常类型分布:")
    anomaly_types = df[df['label'] != 0]['anomaly_type'].value_counts().sort_index()
    for anomaly_type, count in anomaly_types.items():
        print(f"  {anomaly_type}: {count}")
    
    print(f"\n总计异常样本: {len(df[df['label'] != 0])}")
    print(f"正常样本: {len(df[df['label'] == 0])}")
    
except Exception as e:
    print(f"❌ 读取训练数据失败: {e}")

print()

# 3. 测试一个简单的预测
try:
    # 创建一个简单的测试特征
    test_features = np.array([[75.0, 1.5, 11.75, 0.005, 12.0, 35.0]])
    
    print("🧪 测试预测:")
    print(f"测试特征: {test_features[0]}")
    
    if model_data is None:
        print("❌ 无法测试，分类器加载失败")
    elif isinstance(model_data, dict):
        model = model_data.get('model')
        encoder = model_data.get('label_encoder')
        
        if model and encoder:
            # 预测
            pred_encoded = model.predict(test_features)
            pred_proba = model.predict_proba(test_features)
            
            print(f"编码预测: {pred_encoded}")
            print(f"预测概率: {pred_proba}")
            
            # 解码
            pred_label = encoder.inverse_transform(pred_encoded)
            print(f"解码标签: {pred_label}")
            print(f"编码器类别: {encoder.classes_}")
        else:
            print("❌ 模型或编码器不存在")
    else:
        print("❌ 无法测试，模型格式不是字典")
        
except Exception as e:
    print(f"❌ 测试预测失败: {e}")

print("=" * 50)
print("🎯 调试完成") 