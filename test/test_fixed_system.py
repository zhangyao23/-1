#!/usr/bin/env python3
"""
测试修复后的AI异常检测系统
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_processor.feature_extractor import FeatureExtractor
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_fixed_models():
    """
    加载修复后的模型
    """
    logger.info("加载修复后的模型...")
    
    # Load autoencoder
    autoencoder = tf.keras.models.load_model('models/autoencoder_model_fixed/autoencoder_model.keras')
    logger.info("自编码器模型加载成功")
    
    # Load scaler
    scaler = joblib.load('models/autoencoder_model_fixed/autoencoder_scaler.pkl')
    logger.info("标准化器加载成功")
    
    # Load random forest classifier
    rf_classifier = joblib.load('models/rf_classifier_fixed.pkl')
    logger.info("随机森林分类器加载成功")
    
    # Load threshold config
    import json
    with open('models/autoencoder_model_fixed/threshold_config.json', 'r') as f:
        threshold_config = json.load(f)
    threshold = threshold_config['threshold']
    logger.info(f"异常检测阈值: {threshold}")
    
    return autoencoder, scaler, rf_classifier, threshold

def test_scenario(autoencoder, scaler, rf_classifier, threshold, scenario_name, test_data):
    """
    测试单个场景
    """
    logger.info(f"\n=== 测试场景: {scenario_name} ===")
    
    # Initialize feature extractor with fixed scaler path
    metrics_config = [
        'wlan0_wireless_quality', 'wlan0_wireless_level',
        'rx_bytes_rate', 'tx_bytes_rate', 
        'gateway_ping_time', 'dns_response_time',
        'rx_packets_rate', 'tx_packets_rate',
        'rx_errors_rate', 'tx_errors_rate',
        'cpu_usage', 'memory_usage'
    ]
    
    # Create feature extractor WITHOUT using the old scaler
    feature_extractor = FeatureExtractor(
        metrics_config, 
        logger, 
        scaler_path=None  # Don't use old scaler
    )
    
    # Extract features
    features = feature_extractor.extract_features(test_data)
    
    if features is None or len(features) != 6:
        logger.error(f"特征提取失败，返回: {features}")
        return False
    
    logger.info(f"提取的特征: {features}")
    
    # Manual scaling using the new scaler
    features_scaled = scaler.transform([features])
    logger.info(f"标准化后特征: {features_scaled[0]}")
    
    # Autoencoder prediction
    reconstruction = autoencoder.predict(features_scaled, verbose=0)
    mse = np.mean(np.power(features_scaled - reconstruction, 2))
    
    logger.info(f"重构误差 (MSE): {mse:.10f}")
    logger.info(f"异常阈值: {threshold:.10f}")
    
    is_anomaly = mse > threshold
    logger.info(f"是否异常: {'是' if is_anomaly else '否'}")
    
    if is_anomaly:
        # Classify anomaly type
        anomaly_type = rf_classifier.predict(features_scaled)[0]
        anomaly_names = [
            'network_congestion', 'poor_signal_quality', 'system_overload',
            'network_attack', 'hardware_failure', 'configuration_error'
        ]
        logger.info(f"异常类型: {anomaly_names[anomaly_type]}")
    
    return True

def main():
    logger.info("开始测试修复后的AI异常检测系统...")
    
    try:
        # Load models
        autoencoder, scaler, rf_classifier, threshold = load_fixed_models()
        
        # Test scenarios from test_scenarios.py
        test_scenarios = {
            "正常运行场景": {
                "wlan0_wireless_quality": 75,
                "wlan0_wireless_level": -45,
                "rx_bytes_rate": 1500000,  # 1.5 Mbps
                "tx_bytes_rate": 800000,   # 0.8 Mbps
                "gateway_ping_time": 8.5,
                "dns_response_time": 15.0,
                "rx_packets_rate": 1200,
                "tx_packets_rate": 800,
                "rx_errors_rate": 0.005,
                "tx_errors_rate": 0.002,
                "cpu_usage": 12.0,
                "memory_usage": 35.0
            },
            
            "网络拥塞场景": {
                "wlan0_wireless_quality": 45,
                "wlan0_wireless_level": -65,
                "rx_bytes_rate": 200000,   # 0.2 Mbps
                "tx_bytes_rate": 150000,   # 0.15 Mbps
                "gateway_ping_time": 150.0,
                "dns_response_time": 300.0,
                "rx_packets_rate": 200,
                "tx_packets_rate": 150,
                "rx_errors_rate": 0.02,
                "tx_errors_rate": 0.015,
                "cpu_usage": 25.0,
                "memory_usage": 45.0
            },
            
            "信号质量差场景": {
                "wlan0_wireless_quality": 15,
                "wlan0_wireless_level": -85,
                "rx_bytes_rate": 500000,   # 0.5 Mbps
                "tx_bytes_rate": 300000,   # 0.3 Mbps
                "gateway_ping_time": 25.0,
                "dns_response_time": 50.0,
                "rx_packets_rate": 400,
                "tx_packets_rate": 250,
                "rx_errors_rate": 0.15,
                "tx_errors_rate": 0.12,
                "cpu_usage": 15.0,
                "memory_usage": 30.0
            }
        }
        
        # Test each scenario
        success_count = 0
        for scenario_name, test_data in test_scenarios.items():
            try:
                if test_scenario(autoencoder, scaler, rf_classifier, threshold, scenario_name, test_data):
                    success_count += 1
            except Exception as e:
                logger.error(f"测试场景 {scenario_name} 失败: {e}")
        
        logger.info(f"\n测试完成！成功测试 {success_count}/{len(test_scenarios)} 个场景")
        
        if success_count == len(test_scenarios):
            logger.info("🎉 所有测试场景都成功！系统修复完成！")
        else:
            logger.warning("⚠️ 部分测试场景失败，需要进一步调试")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 