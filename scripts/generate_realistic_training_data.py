#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于真实网络指标的训练数据生成器

根据实际的11个网络指标生成训练数据，确保模型训练与实际测试数据格式一致：
1. 基于真实网络指标的正常数据分布
2. 模拟各种网络异常场景的异常数据
3. 确保数据分布更接近实际网络环境
"""

import os
import csv
import json
import numpy as np
import random
from typing import Dict, List, Any

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
REALISTIC_NORMAL_FILE = os.path.join(DATA_DIR, 'realistic_normal_traffic.csv')
REALISTIC_ANOMALIES_FILE = os.path.join(DATA_DIR, 'realistic_labeled_anomalies.csv')

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 真实网络指标的定义
REAL_NETWORK_METRICS = [
    'wlan0_wireless_quality',      # WiFi信号质量 (0-100)
    'wlan0_signal_level',          # WiFi信号强度 (通常负值dBm)
    'wlan0_noise_level',           # WiFi噪声水平 (通常负值dBm)
    'wlan0_rx_packets',            # 接收数据包数量
    'wlan0_tx_packets',            # 发送数据包数量
    'wlan0_rx_bytes',              # 接收字节数
    'wlan0_tx_bytes',              # 发送字节数
    'gateway_ping_time',           # 网关Ping延迟 (ms)
    'dns_resolution_time',         # DNS解析时间 (ms)
    'memory_usage_percent',        # 内存使用率 (0-100%)
    'cpu_usage_percent'            # CPU使用率 (0-100%)
]

# 正常网络环境的典型值范围
NORMAL_RANGES = {
    'wlan0_wireless_quality': (60, 100),        # 良好的WiFi质量
    'wlan0_signal_level': (-65, -30),           # 良好的信号强度
    'wlan0_noise_level': (-90, -70),            # 低噪声
    'wlan0_rx_packets': (50, 200),              # 正常包数量
    'wlan0_tx_packets': (30, 150),              # 正常包数量
    'wlan0_rx_bytes': (1024, 10240),            # 正常字节数
    'wlan0_tx_bytes': (512, 8192),              # 正常字节数
    'gateway_ping_time': (1, 20),               # 良好的延迟
    'dns_resolution_time': (5, 50),             # 正常DNS响应
    'memory_usage_percent': (20, 70),           # 正常内存使用
    'cpu_usage_percent': (5, 40)                # 正常CPU使用
}

# 异常场景定义
ANOMALY_SCENARIOS = {
    'signal_degradation': {
        'description': 'WiFi信号衰减',
        'modifications': {
            'wlan0_wireless_quality': (0, 40),
            'wlan0_signal_level': (-90, -70),
            'gateway_ping_time': (50, 200)
        }
    },
    'network_congestion': {
        'description': '网络拥塞',
        'modifications': {
            'gateway_ping_time': (100, 500),
            'dns_resolution_time': (100, 300),
            'wlan0_rx_packets': (500, 2000),
            'wlan0_tx_packets': (400, 1800)
        }
    },
    'high_interference': {
        'description': '高干扰环境',
        'modifications': {
            'wlan0_noise_level': (-60, -40),
            'wlan0_wireless_quality': (10, 50),
            'wlan0_signal_level': (-80, -60)
        }
    },
    'resource_overload': {
        'description': '系统资源过载',
        'modifications': {
            'memory_usage_percent': (80, 95),
            'cpu_usage_percent': (70, 95),
            'gateway_ping_time': (30, 100)
        }
    },
    'connection_issues': {
        'description': '连接问题',
        'modifications': {
            'gateway_ping_time': (200, 1000),
            'dns_resolution_time': (200, 800),
            'wlan0_rx_packets': (0, 20),
            'wlan0_tx_packets': (0, 15)
        }
    },
    'bandwidth_saturation': {
        'description': '带宽饱和',
        'modifications': {
            'wlan0_rx_bytes': (50000, 100000),
            'wlan0_tx_bytes': (40000, 90000),
            'wlan0_rx_packets': (1000, 5000),
            'wlan0_tx_packets': (800, 4000)
        }
    },
    'mixed_anomaly': {
        'description': '混合异常',
        'modifications': {
            'wlan0_wireless_quality': (0, 30),
            'memory_usage_percent': (85, 98),
            'gateway_ping_time': (100, 300),
            'dns_resolution_time': (100, 400)
        }
    }
}

def generate_normal_sample() -> Dict[str, float]:
    """生成一个正常网络状态的数据样本"""
    sample = {}
    
    for metric in REAL_NETWORK_METRICS:
        min_val, max_val = NORMAL_RANGES[metric]
        
        # 添加一些相关性模拟真实网络行为
        if metric == 'wlan0_wireless_quality':
            # WiFi质量作为基础参考
            base_quality = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 6)
            sample[metric] = max(min_val, min(max_val, base_quality))
        
        elif metric == 'wlan0_signal_level':
            # 信号强度与质量相关
            quality = sample.get('wlan0_wireless_quality', 80)
            # 质量越高，信号强度越好（数值越接近0）
            signal_base = -30 - (100 - quality) * 0.35
            sample[metric] = max(min_val, min(max_val, signal_base + np.random.normal(0, 5)))
        
        elif metric == 'gateway_ping_time':
            # Ping时间与WiFi质量负相关
            quality = sample.get('wlan0_wireless_quality', 80)
            ping_base = max_val - (quality - min_val) / (100 - min_val) * (max_val - min_val)
            sample[metric] = max(min_val, ping_base + np.random.normal(0, 3))
        
        elif metric in ['wlan0_rx_packets', 'wlan0_tx_packets']:
            # 数据包数量有一定相关性
            base_packets = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 6)
            sample[metric] = max(min_val, min(max_val, base_packets))
        
        elif metric in ['wlan0_rx_bytes', 'wlan0_tx_bytes']:
            # 字节数与包数量相关
            packets_key = metric.replace('_bytes', '_packets')
            if packets_key in sample:
                avg_packet_size = np.random.uniform(50, 200)  # 平均包大小
                base_bytes = sample[packets_key] * avg_packet_size
                sample[metric] = max(min_val, min(max_val, base_bytes))
            else:
                sample[metric] = np.random.uniform(min_val, max_val)
        
        elif metric in ['memory_usage_percent', 'cpu_usage_percent']:
            # 系统资源使用有一定相关性
            base_usage = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 8)
            sample[metric] = max(min_val, min(max_val, base_usage))
        
        else:
            # 其他指标使用正态分布
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6
            sample[metric] = max(min_val, min(max_val, np.random.normal(mean, std)))
    
    return sample

def generate_anomaly_sample(scenario_name: str) -> Dict[str, Any]:
    """生成指定异常场景的数据样本"""
    if scenario_name not in ANOMALY_SCENARIOS:
        raise ValueError(f"未知的异常场景: {scenario_name}")
    
    scenario = ANOMALY_SCENARIOS[scenario_name]
    sample = generate_normal_sample()  # 从正常样本开始
    
    # 应用异常修改
    for metric, (min_val, max_val) in scenario['modifications'].items():
        if metric in sample:
            # 在异常范围内生成值
            sample[metric] = np.random.uniform(min_val, max_val)
    
    # 添加标签
    sample['label'] = scenario_name
    
    return sample

def generate_realistic_normal_data(num_samples: int = 15000):
    """生成基于真实网络指标的正常数据"""
    print(f"🔄 生成 {num_samples} 条基于真实网络指标的正常数据...")
    
    normal_samples = []
    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"  进度: {i + 1}/{num_samples}")
        
        sample = generate_normal_sample()
        normal_samples.append([sample[metric] for metric in REAL_NETWORK_METRICS])
    
    # 保存到CSV文件
    with open(REALISTIC_NORMAL_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(REAL_NETWORK_METRICS)  # 使用真实指标名称作为表头
        writer.writerows(normal_samples)
    
    print(f"✅ 正常数据已保存到: {REALISTIC_NORMAL_FILE}")
    return normal_samples

def generate_realistic_anomaly_data(samples_per_scenario: int = 300):
    """生成基于真实网络指标的异常数据"""
    print(f"🔄 生成异常数据，每个场景 {samples_per_scenario} 个样本...")
    
    anomaly_samples = []
    
    for scenario_name, scenario_info in ANOMALY_SCENARIOS.items():
        print(f"  生成场景: {scenario_name} - {scenario_info['description']}")
        
        for i in range(samples_per_scenario):
            sample = generate_anomaly_sample(scenario_name)
            row = [sample[metric] for metric in REAL_NETWORK_METRICS] + [sample['label']]
            anomaly_samples.append(row)
    
    # 随机打乱数据
    np.random.shuffle(anomaly_samples)
    
    # 保存到CSV文件
    with open(REALISTIC_ANOMALIES_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(REAL_NETWORK_METRICS + ['label'])
        writer.writerows(anomaly_samples)
    
    total_anomalies = len(anomaly_samples)
    print(f"✅ 异常数据已保存到: {REALISTIC_ANOMALIES_FILE}")
    print(f"   总计 {total_anomalies} 条异常数据，{len(ANOMALY_SCENARIOS)} 个场景")
    
    return anomaly_samples

def analyze_data_distribution():
    """分析生成的数据分布"""
    print("\n📊 数据分布分析:")
    
    # 分析正常数据
    if os.path.exists(REALISTIC_NORMAL_FILE):
        normal_data = []
        with open(REALISTIC_NORMAL_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                normal_data.append({k: float(v) for k, v in row.items()})
        
        print(f"✅ 正常数据: {len(normal_data)} 条")
        
        # 显示每个指标的统计信息
        for metric in REAL_NETWORK_METRICS[:5]:  # 只显示前5个避免输出过多
            values = [sample[metric] for sample in normal_data]
            print(f"   {metric}: 均值={np.mean(values):.2f}, 标准差={np.std(values):.2f}")
    
    # 分析异常数据
    if os.path.exists(REALISTIC_ANOMALIES_FILE):
        anomaly_data = []
        with open(REALISTIC_ANOMALIES_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                anomaly_data.append(row)
        
        print(f"✅ 异常数据: {len(anomaly_data)} 条")
        
        # 按场景统计
        scenario_counts = {}
        for sample in anomaly_data:
            label = sample['label']
            scenario_counts[label] = scenario_counts.get(label, 0) + 1
        
        for scenario, count in scenario_counts.items():
            print(f"   {scenario}: {count} 条")

def main():
    """主函数"""
    print("🚀 开始生成基于真实网络指标的训练数据")
    print("="*60)
    
    # 生成正常数据
    normal_samples = generate_realistic_normal_data(15000)
    
    # 生成异常数据
    anomaly_samples = generate_realistic_anomaly_data(300)
    
    # 分析数据分布
    analyze_data_distribution()
    
    print("\n🎯 数据生成完成!")
    print("新数据特点:")
    print("✅ 基于11个真实网络指标")
    print("✅ 正常数据模拟真实网络环境")
    print("✅ 异常数据覆盖7种常见网络问题")
    print("✅ 数据分布更接近实际使用场景")
    print("\n📋 下一步建议:")
    print("1. 使用新数据重新训练模型:")
    print("   python scripts/train_models.py --data realistic")
    print("2. 验证模型在真实数据上的表现")
    
if __name__ == "__main__":
    main() 