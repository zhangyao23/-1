#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试自编码器推理过程
分析为什么不同输入产生相同的重构误差
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
import json
from src.ai_models.autoencoder_model import AutoencoderModel
from src.logger.system_logger import SystemLogger

def load_config():
    """加载系统配置"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def debug_autoencoder_inference():
    """调试自编码器推理过程"""
    
    print("🔍 调试自编码器推理过程")
    print("="*50)
    
    try:
        # 加载配置
        config = load_config()
        logger = SystemLogger(config['logging'])
        logger.set_log_level('WARNING')  # 减少日志输出
        
        # 初始化自编码器
        autoencoder_config = config['ai_models']['autoencoder']
        autoencoder = AutoencoderModel(autoencoder_config, logger)
        
        # 加载scaler
        scaler_path = os.path.join(autoencoder_config['model_path'], 'autoencoder_scaler.pkl')
        scaler = joblib.load(scaler_path)
        
        print(f"✅ 模型路径: {autoencoder_config['model_path']}")
        print(f"✅ Scaler路径: {scaler_path}")
        print(f"✅ 异常阈值: {autoencoder_config['threshold']}")
        
        # 测试不同的输入场景
        test_scenarios = {
            "正常场景": {
                "原始特征": [75.0, 1.5, 11.75, 0.005, 12.0, 35.0],
                "期望": "低重构误差"
            },
            "信号异常": {
                "原始特征": [25.0, 0.15, 150.0, 0.15, 85.0, 90.0],
                "期望": "高重构误差"
            },
            "全零测试": {
                "原始特征": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "期望": "基准误差"
            },
            "全一测试": {
                "原始特征": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "期望": "不同误差"
            }
        }
        
        print("\n📊 测试不同输入场景:")
        print("-" * 50)
        
        for scenario_name, scenario_data in test_scenarios.items():
            print(f"\n🧪 {scenario_name}:")
            
            # 原始6维特征
            features = np.array(scenario_data["原始特征"])
            print(f"   原始特征: {features}")
            
            # 标准化前的分析
            print(f"   特征范围: [{features.min():.2f}, {features.max():.2f}]")
            print(f"   特征标准差: {features.std():.4f}")
            
            # 应用标准化
            features_2d = features.reshape(1, -1)
            features_scaled = scaler.transform(features_2d)
            features_scaled_1d = features_scaled.flatten()
            
            print(f"   标准化后: {features_scaled_1d}")
            print(f"   标准化范围: [{features_scaled_1d.min():.4f}, {features_scaled_1d.max():.4f}]")
            print(f"   标准化标准差: {features_scaled_1d.std():.4f}")
            
            # 调用自编码器
            result = autoencoder.predict(features_scaled)
            
            # 分析结果
            recon_error = result.get('reconstruction_error', 0.0)
            is_anomaly = result.get('is_anomaly', False)
            
            print(f"   ➤ 重构误差: {recon_error:.6f}")
            print(f"   ➤ 是否异常: {is_anomaly}")
            print(f"   ➤ 期望结果: {scenario_data['期望']}")
        
        print("\n🔍 Scaler统计信息:")
        print("-" * 30)
        print(f"   Scaler类型: {type(scaler).__name__}")
        
        # RobustScaler的属性与StandardScaler不同
        if hasattr(scaler, 'center_'):
            print(f"   Center (中位数): {scaler.center_}")
        if hasattr(scaler, 'scale_'):
            print(f"   Scale (IQR): {scaler.scale_}")
        if hasattr(scaler, 'mean_'):
            print(f"   Mean: {scaler.mean_}")
        
        # 检查scaler是否合理
        if hasattr(scaler, 'scale_') and np.any(scaler.scale_ == 0):
            print("   ⚠️  警告: 某些特征的标准差为0，可能导致数值问题！")
            zero_scale_indices = np.where(scaler.scale_ == 0)[0]
            print(f"   零标准差特征索引: {zero_scale_indices}")
        
        # 关键问题：分析为什么自编码器总是输出0.000160
        print("\n🚨 关键问题分析:")
        print("-" * 30)
        print("   ❌ 所有输入都产生相同重构误差: 0.000160")
        print("   ❌ 这表明自编码器模型有严重缺陷")
        print("   ➤ 可能原因:")
        print("      1. 模型训练失败或过拟合")
        print("      2. 模型权重全为零或常数")
        print("      3. 模型保存/加载有问题")
        print("      4. SavedModel推理函数异常")
        
    except Exception as e:
        print(f"❌ 调试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("🎯 调试完成")

if __name__ == "__main__":
    debug_autoencoder_inference() 