#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging

def create_improved_autoencoder(input_dim):
    """创建改进的深度自编码器"""
    input_layer = Input(shape=(input_dim,))
    
    # 编码器 (6 -> 8 -> 4)
    encoded = Dense(8, activation='relu')(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(4, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    # 解码器 (4 -> 8 -> 6)
    decoded = Dense(8, activation='relu')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder

def main():
    print("开始训练改进的自编码器模型...")
    
    # 加载数据
    try:
        df = pd.read_csv('data/improved_training_data_6d.csv')
        print(f"加载数据成功: 总共 {len(df)} 条样本")
        
        # 分离正常和异常数据
        normal_data = df[df['label'] == 0]
        anomaly_data = df[df['label'] != 0]
        print(f"正常数据: {len(normal_data)} 条, 异常数据: {len(anomaly_data)} 条")
    except FileNotFoundError:
        print("错误: 找不到改进的训练数据文件")
        return
    
    # 提取特征
    feature_columns = [
        'avg_signal_strength', 'avg_data_rate', 'avg_latency',
        'packet_loss_rate', 'system_load', 'network_stability'
    ]
    
    # 训练自编码器（仅使用正常数据）
    X_normal = normal_data[feature_columns].values
    
    # 数据预处理
    scaler = RobustScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    # 创建和训练自编码器
    autoencoder = create_improved_autoencoder(6)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    print("开始训练自编码器...")
    history = autoencoder.fit(
        X_normal_scaled, X_normal_scaled,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 计算异常阈值
    X_pred = autoencoder.predict(X_normal_scaled)
    reconstruction_errors = np.mean(np.square(X_normal_scaled - X_pred), axis=1)
    threshold = np.percentile(reconstruction_errors, 95)
    
    print(f"异常检测阈值: {threshold:.6f}")
    
    # 创建保存目录
    model_dir = 'models/autoencoder_model_improved'
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存自编码器模型为SavedModel格式（兼容测试工具）
    autoencoder.export(model_dir)
    print(f"自编码器模型已保存到: {model_dir}")
    
    # 同时保存为.keras格式作为备份
    autoencoder.save(f'{model_dir}/autoencoder_model.keras')
    print(f"备份模型已保存为: {model_dir}/autoencoder_model.keras")
    
    # 保存配置文件
    config = {
        'threshold': float(threshold),
        'input_features': 6,
        'encoding_dim': 4,
        'feature_columns': feature_columns,
        'model_version': 'improved_v2.0'
    }
    
    with open(f'{model_dir}/autoencoder_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 保存标准化器
    joblib.dump(scaler, f'{model_dir}/autoencoder_scaler.pkl')
    
    # 训练分类器
    print("\n开始训练改进的分类器...")
    
    # 准备分类数据 - 只对异常数据进行分类训练
    X_anomaly = anomaly_data[feature_columns].values
    y_anomaly = list(anomaly_data['anomaly_type'])
    
    # 数据预处理
    X_anomaly_scaled = scaler.transform(X_anomaly)
    
    # 训练分类器
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_anomaly_scaled, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly
    )
    
    rf_classifier.fit(X_train, y_train)
    
    # 评估分类器
    y_pred = rf_classifier.predict(X_test)
    print("\n分类器性能报告:")
    print(classification_report(y_test, y_pred))
    
    # 保存分类器
    joblib.dump(rf_classifier, 'models/rf_classifier_improved.pkl')
    print("分类器已保存到: models/rf_classifier_improved.pkl")
    
    # 测试完整系统
    print("\n测试完整系统...")
    
    # 测试正常数据
    normal_sample = X_normal_scaled[:10]
    normal_pred = autoencoder.predict(normal_sample)
    normal_errors = np.mean(np.square(normal_sample - normal_pred), axis=1)
    
    print(f"正常数据重构误差 (前10个): {normal_errors}")
    print(f"异常阈值: {threshold:.6f}")
    print(f"正常数据异常检测结果: {(normal_errors > threshold).sum()}/10 被误判为异常")
    
    # 测试异常数据
    if len(anomaly_data) > 0:
        test_count = min(10, len(anomaly_data))
        anomaly_sample = scaler.transform(anomaly_data[feature_columns].values[:test_count])
        anomaly_pred = autoencoder.predict(anomaly_sample)
        anomaly_errors = np.mean(np.square(anomaly_sample - anomaly_pred), axis=1)
        
        print(f"异常数据重构误差 (前{test_count}个): {anomaly_errors}")
        print(f"异常数据异常检测结果: {(anomaly_errors > threshold).sum()}/{test_count} 被正确识别为异常")
    
    print("\n✅ 改进模型训练完成!")

if __name__ == "__main__":
    main() 