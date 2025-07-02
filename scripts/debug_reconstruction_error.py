#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试自编码器重构误差的脚本
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger.system_logger import SystemLogger
from src.feature_processor.feature_extractor import FeatureExtractor
from src.ai_models.autoencoder_model import AutoencoderModel

def test_normal_scenario():
    """测试正常场景的重构误差"""
    try:
        # 加载配置
        config_path = os.path.join(project_root, 'config', 'system_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建简单的日志对象
        class SimpleLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): print(f"DEBUG: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
        
        logger = SimpleLogger()
        
        # 创建正常场景数据
        normal_data = {
            'wlan0_wireless_quality': 85.0,
            'wlan0_signal_level': -45.0,
            'wlan0_noise_level': -90.0,
            'wlan0_rx_packets': 50000,
            'wlan0_tx_packets': 30000,
            'wlan0_rx_bytes': 75000000,
            'wlan0_tx_bytes': 25000000,
            'gateway_ping_time': 8.5,
            'dns_resolution_time': 15.2,
            'memory_usage_percent': 35.0,
            'cpu_usage_percent': 12.0
        }
        
        print("🔍 调试正常场景的重构误差")
        print("=" * 50)
        print(f"输入数据: {normal_data}")
        
        # 初始化特征提取器
        real_metrics = list(normal_data.keys())
        scaler_path = os.path.join(config['ai_models']['autoencoder']['model_path'], 'autoencoder_scaler.pkl')
        feature_extractor = FeatureExtractor(real_metrics, logger, scaler_path=scaler_path)
        
        # 特征提取
        features = feature_extractor.extract_features(normal_data)
        print(f"提取的特征维度: {len(features)}")
        print(f"特征值范围: [{np.min(features):.6f}, {np.max(features):.6f}]")
        print(f"特征值: {features}")
        
        # 初始化自编码器
        autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
        
        # 预测
        result = autoencoder.predict(features)
        
        print("\n📊 自编码器分析结果:")
        print(f"重构误差: {result['reconstruction_error']:.6f}")
        print(f"当前阈值: {result['threshold']:.6f}")
        print(f"误差/阈值比: {result['reconstruction_error']/result['threshold']:.6f}")
        print(f"是否异常: {result['is_anomaly']}")
        print(f"置信度: {result['confidence']:.6f}")
        print(f"异常得分: {result['anomaly_score']:.6f}")
        
        # 如果误差很大，分析原因
        if result['reconstruction_error'] > result['threshold']:
            print("\n⚠️ 分析: 正常数据被误报为异常")
            print("可能原因:")
            print("1. 特征标准化问题")
            print("2. 模型训练数据与测试数据分布不一致")
            print("3. 阈值设置不合理")
            
            # 建议阈值
            suggested_threshold = result['reconstruction_error'] * 1.2
            print(f"建议阈值: {suggested_threshold:.6f} (当前误差 × 1.2)")
        
        return result
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_normal_scenario() 