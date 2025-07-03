#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重新训练自编码器模型
解决当前模型输出相同重构误差的问题
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import logging

def create_improved_autoencoder(input_dim=6, encoding_dim=4):
    """创建改进的深度自编码器"""
    print(f"创建自编码器: 输入维度={input_dim}, 编码维度={encoding_dim}")
    
    # 输入层
    input_layer = Input(shape=(input_dim,), name='input')
    
    # 编码器 (6 -> 8 -> 4)
    encoded = Dense(8, activation='relu', name='encoder_1')(input_layer)
    encoded = BatchNormalization(name='encoder_bn1')(encoded)
    encoded = Dropout(0.1, name='encoder_dropout1')(encoded)
    
    encoded = Dense(encoding_dim, activation='relu', name='encoder_2')(encoded)
    encoded = BatchNormalization(name='encoder_bn2')(encoded)
    
    # 解码器 (4 -> 8 -> 6)
    decoded = Dense(8, activation='relu', name='decoder_1')(encoded)
    decoded = BatchNormalization(name='decoder_bn1')(decoded)
    decoded = Dropout(0.1, name='decoder_dropout1')(decoded)
    
    decoded = Dense(input_dim, activation='linear', name='decoder_output')(decoded)
    
    # 创建自编码器模型
    autoencoder = Model(input_layer, decoded, name='autoencoder')
    
    # 编译模型
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder

def load_and_prepare_data():
    """加载和准备训练数据"""
    print("加载训练数据...")
    
    # 加载数据
    df = pd.read_csv('data/improved_training_data_6d.csv')
    print(f"总数据量: {len(df)}")
    
    # 只使用正常数据进行无监督训练
    normal_data = df[df['label'] == 0]
    print(f"正常数据量: {len(normal_data)}")
    
    # 提取6维特征
    feature_columns = [
        'avg_signal_strength', 'avg_data_rate', 'avg_latency',
        'packet_loss_rate', 'system_load', 'network_stability'
    ]
    
    X_normal = normal_data[feature_columns].values
    print(f"特征形状: {X_normal.shape}")
    
    # 检查数据质量
    print("\\n数据质量检查:")
    print(f"特征统计:")
    for i, col in enumerate(feature_columns):
        values = X_normal[:, i]
        print(f"  {col}: [{values.min():.3f}, {values.max():.3f}], std={values.std():.3f}")
    
    # 数据标准化
    print("\\n进行数据标准化...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_normal)
    
    print("标准化后的数据统计:")
    for i, col in enumerate(feature_columns):
        values = X_scaled[:, i]
        print(f"  {col}: [{values.min():.3f}, {values.max():.3f}], std={values.std():.3f}")
    
    # 分割训练和验证集
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
    print(f"\\n训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    
    return X_train, X_val, scaler, feature_columns

def train_autoencoder(X_train, X_val):
    """训练自编码器"""
    print("\\n开始训练自编码器...")
    
    # 创建模型
    autoencoder = create_improved_autoencoder()
    
    # 打印模型架构
    print("\\n模型架构:")
    autoencoder.summary()
    
    # 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 训练模型
    history = autoencoder.fit(
        X_train, X_train,  # 自编码器的目标是重构输入
        epochs=200,
        batch_size=64,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return autoencoder, history

def evaluate_model(autoencoder, X_train, X_val, scaler):
    """评估模型性能"""
    print("\\n=== 模型评估 ===")
    
    # 在训练集上评估
    train_pred = autoencoder.predict(X_train)
    train_mse = mean_squared_error(X_train, train_pred)
    train_errors = np.mean(np.square(X_train - train_pred), axis=1)
    
    # 在验证集上评估
    val_pred = autoencoder.predict(X_val)
    val_mse = mean_squared_error(X_val, val_pred)
    val_errors = np.mean(np.square(X_val - val_pred), axis=1)
    
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"验证集 MSE: {val_mse:.6f}")
    print(f"训练集重构误差范围: [{train_errors.min():.6f}, {train_errors.max():.6f}]")
    print(f"验证集重构误差范围: [{val_errors.min():.6f}, {val_errors.max():.6f}]")
    
    # 计算异常检测阈值（使用95%分位数）
    threshold_95 = np.percentile(train_errors, 95)
    threshold_99 = np.percentile(train_errors, 99)
    
    print(f"\\n建议的异常检测阈值:")
    print(f"  95%分位数: {threshold_95:.6f}")
    print(f"  99%分位数: {threshold_99:.6f}")
    
    # 测试几个不同的输入
    print("\\n=== 差异化测试 ===")
    test_cases = {
        "正常1": np.array([[75.0, 1.5, 11.75, 0.005, 12.0, 35.0]]),
        "正常2": np.array([[70.0, 2.0, 15.0, 0.01, 20.0, 50.0]]),
        "异常1": np.array([[25.0, 0.15, 150.0, 0.15, 85.0, 90.0]]),
        "异常2": np.array([[10.0, 0.1, 200.0, 0.3, 95.0, 95.0]])
    }
    
    for name, test_data in test_cases.items():
        # 标准化测试数据
        test_scaled = scaler.transform(test_data)
        
        # 预测
        pred = autoencoder.predict(test_scaled, verbose=0)
        error = np.mean(np.square(test_scaled - pred))
        
        print(f"  {name}: 重构误差 = {error:.6f}, 异常={error > threshold_95}")
    
    return threshold_95

def save_model(autoencoder, scaler, threshold):
    """保存模型和相关文件"""
    print("\\n=== 保存模型 ===")
    
    model_dir = "models/autoencoder_model_retrained"
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存为SavedModel格式
    autoencoder.export(model_dir)
    print(f"✅ 自编码器已保存到: {model_dir}")
    
    # 保存scaler
    scaler_path = os.path.join(model_dir, 'autoencoder_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler已保存到: {scaler_path}")
    
    # 保存配置信息
    config = {
        'model_type': 'improved_autoencoder',
        'input_features': 6,
        'encoding_dim': 4,
        'threshold': float(threshold),
        'architecture': '6->8->4->8->6',
        'training_samples': 'normal_data_only',
        'scaler_type': 'RobustScaler'
    }
    
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ 配置已保存到: {config_path}")
    
    return model_dir

def main():
    """主函数"""
    print("🔄 重新训练自编码器")
    print("=" * 50)
    
    try:
        # 1. 加载和准备数据
        X_train, X_val, scaler, feature_columns = load_and_prepare_data()
        
        # 2. 训练模型
        autoencoder, history = train_autoencoder(X_train, X_val)
        
        # 3. 评估模型
        threshold = evaluate_model(autoencoder, X_train, X_val, scaler)
        
        # 4. 保存模型
        model_dir = save_model(autoencoder, scaler, threshold)
        
        print("\\n" + "=" * 50)
        print("🎉 自编码器重训练完成!")
        print(f"📁 模型保存位置: {model_dir}")
        print(f"🎯 建议阈值: {threshold:.6f}")
        print("\\n下一步: 更新配置文件中的模型路径和阈值")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 