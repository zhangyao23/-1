#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特征提取调试脚本

用于诊断特征提取过程中的问题，特别是重构误差异常高的问题。
"""

import os
import sys
import json
import numpy as np

# 将src目录添加到Python路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_processor.feature_extractor import FeatureExtractor
from anomaly_detector.anomaly_engine import AnomalyDetectionEngine
from logger.system_logger import SystemLogger
from ai_models.autoencoder_model import AutoencoderModel
from ai_models.error_classifier import ErrorClassifier

def load_config():
    """加载系统配置"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_default_inputs():
    """获取默认输入数据"""
    inputs_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation_inputs.json')
    try:
        with open(inputs_path, 'r', encoding='utf-8') as f:
            for case in json.load(f):
                if "正常" in case.get("name", ""):
                    return case.get("data", {})
    except (FileNotFoundError, json.JSONDecodeError):
        # 硬编码的后备值
        return {
            'wlan0_wireless_quality': 70.0, 'wlan0_wireless_level': -55.0,
            'wlan0_packet_loss_rate': 0.01, 'wlan0_send_rate_bps': 500000.0,
            'wlan0_recv_rate_bps': 1500000.0, 'tcp_retrans_segments': 5,
            'gateway_ping_time': 12.5, 'dns_response_time': 25.0,
            'tcp_connection_count': 30, 'cpu_percent': 15.0, 'memory_percent': 45.0
        }
    return {}

def debug_feature_extraction():
    """详细调试特征提取过程"""
    print("=== 特征提取调试分析 ===\n")
    
    # 加载配置和初始化组件
    config = load_config()
    logger = SystemLogger(config['logging'])
    logger.set_log_level('DEBUG')
    
    # 获取默认输入数据
    raw_data = get_default_inputs()
    print("1. 原始输入数据:")
    for key, value in raw_data.items():
        print(f"   {key}: {value}")
    print()
    
    # 初始化特征提取器
    extractor = FeatureExtractor(config['data_collection']['metrics'], logger)
    
    print("2. 数据清洗结果:")
    cleaned_data = extractor._clean_raw_data(raw_data)
    for key, value in cleaned_data.items():
        print(f"   {key}: {value}")
    print()
    
    print("3. 基础特征提取:")
    basic_features = extractor._extract_basic_features(cleaned_data)
    for key, value in basic_features.items():
        print(f"   {key}: {value}")
    print()
    
    print("4. 统计特征提取:")
    statistical_features = extractor._calculate_statistical_features(cleaned_data)
    for key, value in statistical_features.items():
        print(f"   {key}: {value}")
    print()
    
    print("5. 时间序列特征提取:")
    temporal_features = extractor._extract_temporal_features(cleaned_data)
    for key, value in temporal_features.items():
        print(f"   {key}: {value}")
    print()
    
    print("6. 合并所有特征:")
    all_features = {**basic_features, **statistical_features, **temporal_features}
    for key, value in all_features.items():
        print(f"   {key}: {value}")
    print()
    
    print("7. 转换为6维向量:")
    feature_vector = extractor._convert_to_vector(all_features)
    feature_names = extractor.get_feature_names()
    print(f"   期望的特征名称: {feature_names}")
    print(f"   6维特征向量: {feature_vector}")
    print(f"   向量维度: {feature_vector.shape}")
    print()
    
    print("8. 检查每个特征的匹配情况:")
    for i, name in enumerate(feature_names):
        value = all_features.get(name, 0.0)
        vector_value = feature_vector[i] if i < len(feature_vector) else 0.0
        print(f"   {name}: 特征字典中={value}, 向量中={vector_value}")
    print()
    
    print("9. 归一化前后对比:")
    print(f"   归一化前: {feature_vector}")
    
    # 检查scaler状态
    scaler_path = "models/autoencoder_model/autoencoder_scaler.pkl"
    print(f"   Scaler文件存在: {os.path.exists(scaler_path)}")
    print(f"   使用预训练scaler: {extractor._use_pretrained_scaler}")
    
    if hasattr(extractor.scaler, 'mean_'):
        print(f"   Scaler均值: {extractor.scaler.mean_}")
        print(f"   Scaler标准差: {extractor.scaler.scale_}")
    else:
        print("   Scaler未拟合")
    
    normalized_vector = extractor._normalize_features(feature_vector.reshape(1, -1))
    print(f"   归一化后: {normalized_vector}")
    print()
    
    print("10. 完整特征提取流程:")
    final_features = extractor.extract_features(raw_data)
    print(f"   最终特征向量: {final_features}")
    print(f"   最终维度: {final_features.shape}")
    print()
    
    return final_features, extractor

def debug_autoencoder():
    """调试自编码器模型"""
    print("=== 自编码器调试分析 ===\n")
    
    config = load_config()
    logger = SystemLogger(config['logging'])
    logger.set_log_level('WARNING')
    
    # 初始化自编码器
    autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
    
    print("1. 自编码器配置:")
    print(f"   输入维度: {config['ai_models']['autoencoder']['input_features']}")
    print(f"   编码维度: {config['ai_models']['autoencoder']['encoding_dim']}")
    print(f"   异常阈值: {config['ai_models']['autoencoder']['anomaly_threshold']}")
    print()
    
    # 获取特征向量
    features, extractor = debug_feature_extraction()
    
    if features.size == 0:
        print("错误：无法获取有效的特征向量")
        return
    
    print("2. 自编码器预测:")
    try:
        # 重构数据
        features_reshaped = features.reshape(1, -1)
        print(f"   输入形状: {features_reshaped.shape}")
        print(f"   输入数据: {features_reshaped}")
        
        reconstructed = autoencoder.model.predict(features_reshaped, verbose=0)
        print(f"   重构形状: {reconstructed.shape}")
        print(f"   重构数据: {reconstructed}")
        
        # 计算重构误差
        mse = np.mean(np.square(features_reshaped - reconstructed))
        print(f"   重构误差 (MSE): {mse}")
        print(f"   异常阈值: {autoencoder.anomaly_threshold}")
        print(f"   是否异常: {mse > autoencoder.anomaly_threshold}")
        
    except Exception as e:
        print(f"   自编码器预测失败: {e}")
    
    print()

def debug_training_data():
    """检查训练数据的特征范围"""
    print("=== 训练数据调试分析 ===\n")
    
    import pandas as pd
    
    # 检查正常数据
    normal_data_path = "data/6d_normal_traffic.csv"
    if os.path.exists(normal_data_path):
        print("1. 正常训练数据统计:")
        df_normal = pd.read_csv(normal_data_path)
        print(f"   数据形状: {df_normal.shape}")
        print("   各特征统计:")
        print(df_normal.describe())
        print()
    else:
        print("   正常训练数据文件不存在")
    
    # 检查异常数据
    anomaly_data_path = "data/6d_labeled_anomalies.csv"
    if os.path.exists(anomaly_data_path):
        print("2. 异常训练数据统计:")
        df_anomaly = pd.read_csv(anomaly_data_path)
        print(f"   数据形状: {df_anomaly.shape}")
        print("   各特征统计:")
        print(df_anomaly.describe())
        print()
    else:
        print("   异常训练数据文件不存在")

def main():
    """主函数"""
    try:
        debug_training_data()
        debug_autoencoder()
        
        print("\n=== 问题诊断 ===")
        print("可能的问题原因:")
        print("1. 特征提取器中某些特征计算不正确")
        print("2. StandardScaler的均值/标准差与训练时不同")
        print("3. 训练数据与测试数据的特征范围差异过大")
        print("4. 自编码器模型与当前特征维度不匹配")
        print("5. 默认输入值可能不在正常范围内")
        
    except Exception as e:
        print(f"调试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 