#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试特征映射过程
分析为什么不同输入产生相同的特征向量
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_processor.feature_extractor import FeatureExtractor
import logging

# 设置日志
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def debug_feature_extraction():
    """调试特征提取过程"""
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor([], logger)
    
    # 测试两个明显不同的场景
    scenarios = {
        "正常场景": {
            "wlan0_wireless_quality": 75.0,
            "wlan0_wireless_level": -45.0,
            "wlan0_packet_loss_rate": 0.005,
            "wlan0_send_rate_bps": 1000000.0,
            "wlan0_recv_rate_bps": 2000000.0,
            "tcp_retrans_segments": 2,
            "gateway_ping_time": 8.5,
            "dns_response_time": 15.0,
            "tcp_connection_count": 25,
            "cpu_percent": 12.0,
            "memory_percent": 35.0
        },
        "异常场景": {
            "wlan0_wireless_quality": 25.0,
            "wlan0_wireless_level": -85.0,
            "wlan0_packet_loss_rate": 0.15,
            "wlan0_send_rate_bps": 100000.0,
            "wlan0_recv_rate_bps": 200000.0,
            "tcp_retrans_segments": 15,
            "gateway_ping_time": 120.0,
            "dns_response_time": 180.0,
            "tcp_connection_count": 20,
            "cpu_percent": 85.0,
            "memory_percent": 90.0
        }
    }
    
    for scenario_name, raw_data in scenarios.items():
        print(f"\n{'='*60}")
        print(f"🧪 调试场景: {scenario_name}")
        print(f"{'='*60}")
        
        # 1. 原始数据
        print("\n📊 1. 原始11维输入数据:")
        for key, value in raw_data.items():
            print(f"   {key}: {value}")
        
        # 2. 数据清洗
        cleaned_data = feature_extractor._clean_raw_data(raw_data)
        print("\n🧹 2. 清洗后数据:")
        for key, value in cleaned_data.items():
            print(f"   {key}: {value}")
        
        # 3. 基础特征提取
        basic_features = feature_extractor._extract_basic_features(cleaned_data)
        print("\n🔧 3. 基础特征提取结果:")
        for key, value in basic_features.items():
            print(f"   {key}: {value}")
        
        # 4. 统计特征
        statistical_features = feature_extractor._calculate_statistical_features(cleaned_data)
        print("\n📈 4. 统计特征:")
        for key, value in statistical_features.items():
            print(f"   {key}: {value}")
        
        # 5. 时间序列特征
        temporal_features = feature_extractor._extract_temporal_features(cleaned_data)
        print("\n⏰ 5. 时间序列特征:")
        for key, value in temporal_features.items():
            print(f"   {key}: {value}")
        
        # 6. 合并所有特征
        all_features = {**basic_features, **statistical_features, **temporal_features}
        print("\n🔗 6. 合并后的所有特征:")
        for key, value in all_features.items():
            print(f"   {key}: {value}")
        
        # 7. 转换为6维向量
        feature_vector = feature_extractor._convert_to_vector(all_features)
        print("\n📏 7. 转换为6维向量:")
        feature_names = feature_extractor.get_feature_names()
        for i, (name, value) in enumerate(zip(feature_names, feature_vector)):
            print(f"   {name}: {value}")
        
        # 8. 标准化后的特征
        normalized_features = feature_extractor._normalize_features(feature_vector)
        print("\n⚖️ 8. 标准化后的特征:")
        for i, (name, value) in enumerate(zip(feature_names, normalized_features)):
            print(f"   {name}: {value}")
        
        print(f"\n✅ 最终6维特征向量: {normalized_features}")

if __name__ == "__main__":
    debug_feature_extraction() 