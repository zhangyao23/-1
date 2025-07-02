#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
异常特征分析脚本

分析6种异常类型的特征组合，展示每种异常类型是如何通过多个指标共同定义的
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

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

def create_test_anomaly_samples():
    """创建测试用的异常样本，每种类型都有明显的多指标特征"""
    
    samples = {
        "signal_degradation": {
            "name": "信号衰减异常",
            "description": "WiFi信号质量下降，同时影响多个网络指标",
            "data": {
                "wlan0_wireless_quality": 20.0,      # 信号质量极差
                "wlan0_wireless_level": -85.0,       # 信号强度很弱
                "wlan0_packet_loss_rate": 0.12,      # 丢包率高
                "wlan0_send_rate_bps": 80000.0,      # 发送速率低
                "wlan0_recv_rate_bps": 150000.0,     # 接收速率低
                "tcp_retrans_segments": 18,          # 重传次数多
                "gateway_ping_time": 120.0,          # 延迟高
                "dns_response_time": 250.0,          # DNS慢
                "tcp_connection_count": 15,          # 连接数少
                "cpu_percent": 20.0,                 # CPU正常
                "memory_percent": 40.0               # 内存正常
            },
            "affected_features": ["信号强度", "数据速率", "网络延迟", "丢包率"]
        },
        
        "network_congestion": {
            "name": "网络拥塞异常",
            "description": "网络带宽不足，多个传输指标异常",
            "data": {
                "wlan0_wireless_quality": 65.0,      # 信号质量中等
                "wlan0_wireless_level": -55.0,       # 信号强度中等
                "wlan0_packet_loss_rate": 0.08,      # 丢包率较高
                "wlan0_send_rate_bps": 200000.0,     # 发送速率低
                "wlan0_recv_rate_bps": 350000.0,     # 接收速率低
                "tcp_retrans_segments": 25,          # 重传次数多
                "gateway_ping_time": 80.0,           # 延迟较高
                "dns_response_time": 120.0,          # DNS较慢
                "tcp_connection_count": 45,          # 连接数较多
                "cpu_percent": 25.0,                 # CPU使用稍高
                "memory_percent": 55.0               # 内存使用稍高
            },
            "affected_features": ["数据速率", "网络延迟", "丢包率", "CPU和内存"]
        },
        
        "resource_overload": {
            "name": "资源过载异常", 
            "description": "系统资源不足，影响网络处理能力",
            "data": {
                "wlan0_wireless_quality": 70.0,      # 信号质量正常
                "wlan0_wireless_level": -50.0,       # 信号强度正常
                "wlan0_packet_loss_rate": 0.04,      # 丢包率稍高
                "wlan0_send_rate_bps": 600000.0,     # 发送速率中等
                "wlan0_recv_rate_bps": 900000.0,     # 接收速率中等
                "tcp_retrans_segments": 12,          # 重传次数稍多
                "gateway_ping_time": 35.0,           # 延迟稍高
                "dns_response_time": 60.0,           # DNS稍慢
                "tcp_connection_count": 40,          # 连接数正常
                "cpu_percent": 92.0,                 # CPU使用率极高
                "memory_percent": 94.0               # 内存使用率极高
            },
            "affected_features": ["CPU使用率", "内存使用率", "网络延迟", "数据速率"]
        },
        
        "connection_timeout": {
            "name": "连接超时异常",
            "description": "网络连接不稳定，超时和延迟问题突出",
            "data": {
                "wlan0_wireless_quality": 50.0,      # 信号质量较差
                "wlan0_wireless_level": -70.0,       # 信号强度较弱
                "wlan0_packet_loss_rate": 0.06,      # 丢包率较高
                "wlan0_send_rate_bps": 300000.0,     # 发送速率较低
                "wlan0_recv_rate_bps": 400000.0,     # 接收速率较低
                "tcp_retrans_segments": 15,          # 重传次数较多
                "gateway_ping_time": 300.0,          # 延迟极高
                "dns_response_time": 400.0,          # DNS极慢
                "tcp_connection_count": 10,          # 连接数很少
                "cpu_percent": 30.0,                 # CPU正常
                "memory_percent": 50.0               # 内存正常
            },
            "affected_features": ["网络延迟", "信号强度", "丢包率", "数据速率"]
        },
        
        "packet_corruption": {
            "name": "数据包损坏异常",
            "description": "数据传输质量问题，包损坏和重传频繁",
            "data": {
                "wlan0_wireless_quality": 55.0,      # 信号质量中下
                "wlan0_wireless_level": -65.0,       # 信号强度中下
                "wlan0_packet_loss_rate": 0.10,      # 丢包率高
                "wlan0_send_rate_bps": 400000.0,     # 发送速率中等
                "wlan0_recv_rate_bps": 500000.0,     # 接收速率中等
                "tcp_retrans_segments": 30,          # 重传次数很多
                "gateway_ping_time": 50.0,           # 延迟中等
                "dns_response_time": 90.0,           # DNS中等
                "tcp_connection_count": 25,          # 连接数正常
                "cpu_percent": 22.0,                 # CPU正常
                "memory_percent": 45.0               # 内存正常
            },
            "affected_features": ["丢包率", "信号强度", "数据速率", "重传次数"]
        },
        
        "mixed_anomaly": {
            "name": "混合异常",
            "description": "多种问题同时出现，几乎所有指标都异常",
            "data": {
                "wlan0_wireless_quality": 30.0,      # 信号质量差
                "wlan0_wireless_level": -80.0,       # 信号强度弱
                "wlan0_packet_loss_rate": 0.15,      # 丢包率很高
                "wlan0_send_rate_bps": 100000.0,     # 发送速率很低
                "wlan0_recv_rate_bps": 200000.0,     # 接收速率很低
                "tcp_retrans_segments": 35,          # 重传次数很多
                "gateway_ping_time": 150.0,          # 延迟很高
                "dns_response_time": 300.0,          # DNS很慢
                "tcp_connection_count": 8,           # 连接数很少
                "cpu_percent": 85.0,                 # CPU使用率高
                "memory_percent": 88.0               # 内存使用率高
            },
            "affected_features": ["所有特征", "信号", "网络", "系统资源"]
        }
    }
    
    return samples

def analyze_feature_contributions(sample_data, feature_extractor, engine):
    """分析特征对异常检测的贡献"""
    
    # 提取特征
    feature_vector = feature_extractor.extract_features(sample_data)
    feature_names = feature_extractor.get_feature_names()
    
    if feature_vector.size == 0:
        return None, None
    
    # 进行检测
    is_anomaly, details = engine.detect_anomaly_from_vector(feature_vector, feature_names)
    
    # 分析每个特征的数值
    feature_analysis = {}
    for i, (name, value) in enumerate(zip(feature_names, feature_vector)):
        feature_analysis[name] = {
            'value': float(value),
            'normalized_value': float(value),  # 已经是归一化后的值
            'index': i
        }
    
    return feature_analysis, details

def main():
    """主函数"""
    print("🔍 AI异常检测系统 - 异常特征分析")
    print("="*70)
    print("📋 分析目标：展示每种异常类型如何通过多个指标共同定义")
    print("-"*70)
    
    # 初始化系统组件
    print("⏳ 正在初始化AI检测系统...")
    try:
        config = load_config()
        logger = SystemLogger(config['logging'])
        logger.set_log_level('WARNING')
        
        extractor = FeatureExtractor(config['data_collection']['metrics'], logger)
        autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
        classifier = ErrorClassifier(config['ai_models']['classifier'], logger)
        
        engine = AnomalyDetectionEngine(
            config=config['anomaly_detection'],
            autoencoder=autoencoder, error_classifier=classifier,
            buffer_manager=None, logger=logger
        )
        
        # 校准特征提取器
        normal_baseline = {
            'wlan0_wireless_quality': 70.0, 'wlan0_wireless_level': -55.0,
            'wlan0_packet_loss_rate': 0.01, 'wlan0_send_rate_bps': 500000.0,
            'wlan0_recv_rate_bps': 1500000.0, 'tcp_retrans_segments': 5,
            'gateway_ping_time': 12.5, 'dns_response_time': 25.0,
            'tcp_connection_count': 30, 'cpu_percent': 15.0, 'memory_percent': 45.0
        }
        extractor.extract_features(normal_baseline)
        
        print("✅ 系统初始化完成")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 获取测试样本
    test_samples = create_test_anomaly_samples()
    
    print(f"\n📊 分析 {len(test_samples)} 种异常类型的特征组合")
    print("="*70)
    
    # 分析每种异常类型
    for anomaly_type, sample_info in test_samples.items():
        print(f"\n🧪 {sample_info['name']} ({anomaly_type})")
        print(f"📝 {sample_info['description']}")
        print(f"🎯 主要影响: {', '.join(sample_info['affected_features'])}")
        print("-" * 60)
        
        # 特征分析
        feature_analysis, detection_result = analyze_feature_contributions(
            sample_info['data'], extractor, engine
        )
        
        if feature_analysis is None:
            print("❌ 特征提取失败")
            continue
        
        # 显示检测结果
        if detection_result:
            predicted_type = detection_result.get('predicted_class', 'unknown')
            confidence = detection_result.get('confidence', 0.0)
            print(f"🎯 检测结果: {predicted_type} (置信度: {confidence:.1%})")
            
            # 准确性标记
            if predicted_type == anomaly_type:
                print("✅ 分类正确!")
            else:
                print(f"⚠️  分类结果与预期不符 (预期: {anomaly_type})")
        
        # 显示特征值
        print("\n📊 6维特征向量分析:")
        print("特征名称\t\t\t数值\t\t说明")
        print("-" * 50)
        
        for feature_name, analysis in feature_analysis.items():
            value = analysis['value']
            
            # 判断特征是否异常（基于归一化值）
            if abs(value) > 1.0:
                status = "⚠️ 异常"
            elif abs(value) > 0.5:
                status = "⚠️ 偏高"
            else:
                status = "✅ 正常"
            
            print(f"{feature_name:<24}\t{value:>8.3f}\t\t{status}")
        
        print("\n💡 多指标综合特征:")
        # 统计异常特征数量
        abnormal_features = [name for name, analysis in feature_analysis.items() 
                           if abs(analysis['value']) > 0.5]
        print(f"   异常特征数量: {len(abnormal_features)}/6")
        print(f"   异常特征列表: {', '.join(abnormal_features)}")
        
        print("="*60)
    
    print(f"\n🎯 **核心结论**")
    print("="*70)
    print("✅ **异常类型检测机制：多指标综合判断**")
    print()
    print("1️⃣  **特征组合**: 每种异常类型都通过6个特征的不同组合来识别")
    print("2️⃣  **模式识别**: 随机森林分类器学习多维特征空间中的模式")
    print("3️⃣  **权重分配**: 不同特征对不同异常类型的贡献权重不同")
    print("4️⃣  **容错能力**: 即使单个特征有噪声，多特征组合依然稳定")
    print()
    print("🔍 **6个核心特征维度:**")
    print("   • avg_signal_strength (平均信号强度)")
    print("   • avg_data_rate (平均数据传输速率)")
    print("   • avg_latency (平均网络延迟)")
    print("   • total_packet_loss (总丢包率)")
    print("   • cpu_usage (CPU使用率)")
    print("   • memory_usage (内存使用率)")
    print()
    print("📈 **每种异常类型的特征 'finger print':**")
    print("   • signal_degradation: 信号+速率+延迟")
    print("   • network_congestion: 速率+延迟+丢包")
    print("   • resource_overload: CPU+内存+延迟")
    print("   • connection_timeout: 延迟+信号+丢包")
    print("   • packet_corruption: 丢包+信号+速率")
    print("   • mixed_anomaly: 多特征同时异常")

if __name__ == "__main__":
    main() 