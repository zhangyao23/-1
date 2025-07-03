#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI异常检测训练数据生成器 (使用FeatureExtractor)

使用FeatureExtractor将11个原始网络指标转换为6个工程特征，
然后使用这些工程特征来训练AI模型。

步骤：
1. 生成11个原始网络指标的模拟数据
2. 使用FeatureExtractor转换为6维特征向量
3. 生成标准化的6维特征向量
4. 保存为训练数据

主要步骤：
1. 读取基于真实指标的原始数据
2. 使用FeatureExtractor进行特征工程
3. 生成标准化的6维特征向量
4. 保存用于模型训练的特征数据
"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_processor.feature_extractor import FeatureExtractor
from src.logger.system_logger import SystemLogger

# 数据目录
DATA_DIR = os.path.join(project_root, 'data')
REALISTIC_NORMAL_FILE = os.path.join(DATA_DIR, 'realistic_normal_traffic.csv')
REALISTIC_ANOMALIES_FILE = os.path.join(DATA_DIR, 'realistic_labeled_anomalies.csv')

# 输出文件
PROCESSED_NORMAL_FILE = os.path.join(DATA_DIR, 'processed_normal_traffic.csv')
PROCESSED_ANOMALIES_FILE = os.path.join(DATA_DIR, 'processed_labeled_anomalies.csv')

# 配置文件
CONFIG_FILE = os.path.join(project_root, 'config', 'system_config.json')

def load_config():
    """加载系统配置"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告：无法加载配置文件 {CONFIG_FILE}: {e}")
        # 使用默认配置
        return {
            'logging': {
                'level': 'INFO',
                'file': 'logs/training.log'
            },
            'metrics': [
                'wlan0_wireless_quality', 'wlan0_signal_level', 'wlan0_noise_level',
                'wlan0_rx_packets', 'wlan0_tx_packets', 'wlan0_rx_bytes', 'wlan0_tx_bytes',
                'gateway_ping_time', 'dns_resolution_time', 'memory_usage_percent', 'cpu_usage_percent'
            ]
        }

def convert_to_dict_format(row_data: Dict[str, str]) -> Dict[str, float]:
    """将CSV行数据转换为特征提取器需要的字典格式"""
    converted = {}
    for key, value in row_data.items():
        if key != 'label':  # 跳过标签列
            try:
                converted[key] = float(value)
            except (ValueError, TypeError):
                converted[key] = 0.0
    return converted

def process_normal_data(feature_extractor: FeatureExtractor, logger: SystemLogger):
    """处理正常数据，转换为特征向量"""
    print("🔄 处理正常数据...")
    
    if not os.path.exists(REALISTIC_NORMAL_FILE):
        print(f"❌ 错误：找不到原始正常数据文件 {REALISTIC_NORMAL_FILE}")
        print("请先运行：python scripts/generate_realistic_training_data.py")
        return None
    
    # 读取原始数据
    normal_data = pd.read_csv(REALISTIC_NORMAL_FILE)
    print(f"   原始数据: {len(normal_data)} 条记录，{len(normal_data.columns)} 个指标")
    
    processed_features = []
    feature_names = None
    
    for index, row in normal_data.iterrows():
        if (index + 1) % 1000 == 0:
            print(f"   进度: {index + 1}/{len(normal_data)}")
        
        # 转换为字典格式
        row_dict = convert_to_dict_format(row.to_dict())
        
        # 提取特征
        feature_vector = feature_extractor.extract_features(row_dict)
        
        if len(feature_vector) > 0:
            processed_features.append(feature_vector)
            
            # 获取特征名称（只需要一次）
            if feature_names is None:
                feature_names = feature_extractor.get_feature_names()
    
    if not processed_features:
        print("❌ 错误：未能提取任何特征")
        return None
    
    # 转换为数组
    processed_features = np.array(processed_features)
    print(f"   特征工程完成: {len(processed_features)} 条记录，{processed_features.shape[1]} 个特征")
    
    # 保存处理后的数据
    feature_df = pd.DataFrame(processed_features, columns=feature_names)
    feature_df.to_csv(PROCESSED_NORMAL_FILE, index=False)
    
    print(f"✅ 正常数据特征已保存到: {PROCESSED_NORMAL_FILE}")
    return processed_features, feature_names

def process_anomaly_data(feature_extractor: FeatureExtractor, logger: SystemLogger):
    """处理异常数据，转换为特征向量"""
    print("🔄 处理异常数据...")
    
    if not os.path.exists(REALISTIC_ANOMALIES_FILE):
        print(f"❌ 错误：找不到原始异常数据文件 {REALISTIC_ANOMALIES_FILE}")
        print("请先运行：python scripts/generate_realistic_training_data.py")
        return None
    
    # 读取原始数据
    anomaly_data = pd.read_csv(REALISTIC_ANOMALIES_FILE)
    print(f"   原始数据: {len(anomaly_data)} 条记录")
    
    processed_features = []
    labels = []
    feature_names = None
    
    for index, row in anomaly_data.iterrows():
        if (index + 1) % 200 == 0:
            print(f"   进度: {index + 1}/{len(anomaly_data)}")
        
        # 提取标签
        label = row.get('label', 'unknown')
        labels.append(label)
        
        # 转换为字典格式（排除标签列）
        row_dict = convert_to_dict_format(row.to_dict())
        
        # 提取特征
        feature_vector = feature_extractor.extract_features(row_dict)
        
        if len(feature_vector) > 0:
            processed_features.append(feature_vector)
            
            # 获取特征名称（只需要一次）
            if feature_names is None:
                feature_names = feature_extractor.get_feature_names()
        else:
            # 如果特征提取失败，移除对应的标签
            labels.pop()
    
    if not processed_features:
        print("❌ 错误：未能提取任何特征")
        return None
    
    # 转换为数组
    processed_features = np.array(processed_features)
    print(f"   特征工程完成: {len(processed_features)} 条记录，{processed_features.shape[1]} 个特征")
    
    # 保存处理后的数据
    feature_df = pd.DataFrame(processed_features, columns=feature_names)
    feature_df['label'] = labels
    feature_df.to_csv(PROCESSED_ANOMALIES_FILE, index=False)
    
    print(f"✅ 异常数据特征已保存到: {PROCESSED_ANOMALIES_FILE}")
    
    # 显示标签分布
    label_counts = pd.Series(labels).value_counts()
    print("   异常类别分布:")
    for label, count in label_counts.items():
        print(f"     {label}: {count} 条")
    
    return processed_features, labels, feature_names

def analyze_features(feature_names: List[str]):
    """分析生成的特征"""
    print("\n📊 特征分析:")
    print(f"✅ 总特征数量: {len(feature_names)}")
    print("✅ 特征列表:")
    
    # 按类别分组显示特征
    basic_features = [f for f in feature_names if not any(suffix in f for suffix in ['_trend', '_volatility', '_momentum', '_change_rate', '_mean', '_std', '_median', '_range'])]
    statistical_features = [f for f in feature_names if any(suffix in f for suffix in ['_mean', '_std', '_median', '_range', 'global_'])]
    temporal_features = [f for f in feature_names if any(suffix in f for suffix in ['_trend', '_volatility', '_momentum', '_change_rate'])]
    
    print(f"   基础特征 ({len(basic_features)}): {basic_features[:5]}..." if len(basic_features) > 5 else f"   基础特征: {basic_features}")
    print(f"   统计特征 ({len(statistical_features)}): {statistical_features[:3]}..." if len(statistical_features) > 3 else f"   统计特征: {statistical_features}")
    print(f"   时间特征 ({len(temporal_features)}): {temporal_features[:3]}..." if len(temporal_features) > 3 else f"   时间特征: {temporal_features}")

def main():
    """主函数"""
    print("🚀 开始基于特征工程的训练数据生成")
    print("="*60)
    
    # 加载配置
    config = load_config()
    
    # 初始化日志
    logger = SystemLogger(config['logging'])
    
    # 创建特征提取器（不使用预训练的scaler，用于训练模式）
    print("🔧 初始化特征提取器...")
    # 使用11个真实网络指标配置
    real_metrics = [
        'wlan0_wireless_quality', 'wlan0_signal_level', 'wlan0_noise_level',
        'wlan0_rx_packets', 'wlan0_tx_packets', 'wlan0_rx_bytes', 'wlan0_tx_bytes',
        'gateway_ping_time', 'dns_resolution_time', 'memory_usage_percent', 'cpu_usage_percent'
    ]
    feature_extractor = FeatureExtractor(
        metrics_config=real_metrics,
        logger=logger,
        scaler_path=None  # 不使用预训练scaler
    )
    
    # 处理正常数据
    normal_result = process_normal_data(feature_extractor, logger)
    if normal_result is None:
        print("❌ 正常数据处理失败，退出程序")
        return
    
    normal_features, feature_names = normal_result
    
    # 处理异常数据
    anomaly_result = process_anomaly_data(feature_extractor, logger)
    if anomaly_result is None:
        print("❌ 异常数据处理失败，退出程序")
        return
    
    anomaly_features, anomaly_labels, _ = anomaly_result
    
    # 分析特征
    analyze_features(feature_names)
    
    print("\n🎯 特征工程数据生成完成!")
    print("新数据特点:")
    print("✅ 使用标准特征提取器进行特征工程")
    print("✅ 从11个原始指标生成6个工程特征")
    print("✅ 特征格式与实际测试环境完全一致")
    print("✅ 包含基础、统计和时间序列特征")
    
    print(f"\n📂 输出文件:")
    print(f"   正常数据: {PROCESSED_NORMAL_FILE}")
    print(f"   异常数据: {PROCESSED_ANOMALIES_FILE}")
    
    print("\n📋 下一步建议:")
    print("1. 使用处理后的特征数据重新训练模型:")
    print(f"   python scripts/train_model.py autoencoder --data_path {PROCESSED_NORMAL_FILE}")
    print(f"   python scripts/train_model.py classifier --data_path {PROCESSED_ANOMALIES_FILE}")
    print("2. 验证模型在真实数据上的表现")

if __name__ == "__main__":
    main() 