#!/usr/bin/env python3
"""重新训练分类器脚本"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

print("🔄 重新训练分类器")
print("=" * 60)

# 1. 加载训练数据
print("📚 加载训练数据...")
try:
    df = pd.read_csv('data/improved_training_data_6d.csv')
    print(f"✅ 数据加载成功，共 {len(df)} 个样本")
    
    # 过滤出异常样本（用于训练分类器）
    anomaly_df = df[df['label'] != 0].copy()
    print(f"📊 异常样本数量: {len(anomaly_df)}")
    
    # 检查异常类型分布
    print("\n异常类型分布:")
    type_counts = anomaly_df['anomaly_type'].value_counts().sort_index()
    for anomaly_type, count in type_counts.items():
        print(f"  {anomaly_type}: {count}")
    
except Exception as e:
    print(f"❌ 加载训练数据失败: {e}")
    exit(1)

# 2. 准备特征和标签
print("\n🎯 准备特征和标签...")
try:
    # 提取特征列
    feature_columns = ['avg_signal_strength', 'avg_data_rate', 'avg_latency', 'packet_loss_rate', 'system_load', 'network_stability']
    X = anomaly_df[feature_columns].values
    y = anomaly_df['anomaly_type'].values
    
    print(f"特征维度: {X.shape}")
    print(f"标签数量: {len(y)}")
    print(f"唯一标签: {np.unique(y)}")
    
except Exception as e:
    print(f"❌ 准备数据失败: {e}")
    exit(1)

# 3. 编码标签
print("\n🔤 编码标签...")
try:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"标签映射:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {i}: {class_name}")
    
except Exception as e:
    print(f"❌ 标签编码失败: {e}")
    exit(1)

# 4. 分割训练和测试数据
print("\n✂️ 分割训练测试数据...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
except Exception as e:
    print(f"❌ 数据分割失败: {e}")
    exit(1)

# 5. 训练随机森林分类器
print("\n🌲 训练随机森林分类器...")
try:
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced'
    )
    
    rf_classifier.fit(X_train, y_train)
    print("✅ 模型训练完成")
    
    # 计算训练准确率
    train_score = rf_classifier.score(X_train, y_train)
    print(f"训练准确率: {train_score:.4f}")
    
    # 计算测试准确率
    test_score = rf_classifier.score(X_test, y_test)
    print(f"测试准确率: {test_score:.4f}")
    
except Exception as e:
    print(f"❌ 模型训练失败: {e}")
    exit(1)

# 6. 评估模型
print("\n📊 模型评估...")
try:
    y_pred = rf_classifier.predict(X_test)
    
    print("\n分类报告:")
    target_names = [f"{i}:{name}" for i, name in enumerate(label_encoder.classes_)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
except Exception as e:
    print(f"❌ 模型评估失败: {e}")

# 7. 保存模型
print("\n💾 保存模型...")
try:
    # 确保models目录存在
    os.makedirs('models', exist_ok=True)
    
    # 保存模型和标签编码器
    model_data = {
        'model': rf_classifier,
        'label_encoder': label_encoder,
        'feature_columns': feature_columns,
        'classes': label_encoder.classes_,
        'training_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'features': feature_columns
        }
    }
    
    # 保存主模型
    with open('models/rf_classifier_improved.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ 模型保存成功: models/rf_classifier_improved.pkl")
    
    # 同时保存备份
    with open('models/rf_classifier_backup.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ 备份保存成功: models/rf_classifier_backup.pkl")
    
except Exception as e:
    print(f"❌ 模型保存失败: {e}")
    exit(1)

# 8. 测试加载
print("\n🧪 测试模型加载...")
try:
    with open('models/rf_classifier_improved.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    
    print("✅ 模型加载成功")
    print(f"模型类型: {type(loaded_model)}")
    print(f"包含键: {list(loaded_model.keys())}")
    
    # 测试预测
    test_features = np.array([[75.0, 1.5, 11.75, 0.005, 12.0, 35.0]])
    pred = loaded_model['model'].predict(test_features)
    pred_proba = loaded_model['model'].predict_proba(test_features)
    pred_label = loaded_model['label_encoder'].inverse_transform(pred)
    
    print(f"\n测试预测:")
    print(f"  输入特征: {test_features[0]}")
    print(f"  预测编码: {pred[0]}")
    print(f"  预测标签: {pred_label[0]}")
    print(f"  预测概率: {pred_proba[0]}")
    
except Exception as e:
    print(f"❌ 模型加载测试失败: {e}")

print("\n" + "=" * 60)
print("🎉 分类器重新训练完成!") 