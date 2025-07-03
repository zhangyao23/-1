#!/usr/bin/env python3
"""
单向量异常检测测试脚本
生成随机6维向量并使用训练好的模型进行异常检测
"""

import sys
import os
import json
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_processor.feature_extractor import FeatureExtractor
from anomaly_detector.anomaly_engine import AnomalyDetectionEngine
from logger.system_logger import SystemLogger
from ai_models.autoencoder_model import AutoencoderModel
from ai_models.error_classifier import ErrorClassifier

def load_config():
    """加载系统配置"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'system_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_random_vector():
    """
    生成一个随机的6维向量进行测试
    """
    # 生成接近正常数据分布的向量（大概率是正常的）
    # 基于6维特征: avg_signal_strength, avg_data_rate, avg_latency, packet_loss_rate, system_load, network_stability
    normal_like_vector = np.array([
        np.random.normal(6.25, 2.4),     # avg_signal_strength (0-10范围)
        np.random.normal(-0.04, 0.01),  # avg_data_rate (固定值附近小变化)
        np.random.normal(8.86, 5.0),    # avg_latency (0-50范围)
        np.random.normal(1.41, 0.1),    # packet_loss_rate (1-2范围)
        np.random.normal(-0.32, 0.1),   # system_load (-1到1范围)
        np.random.normal(0.93, 0.05)    # network_stability (0-1范围)
    ])
    
    # 生成明显异常的向量（某些维度有极值）
    anomaly_like_vector = np.array([
        np.random.normal(6.25, 2.4),     # avg_signal_strength
        np.random.normal(-0.04, 0.01),  # avg_data_rate
        np.random.normal(8.86, 5.0),    # avg_latency
        np.random.normal(1.41, 0.1),    # packet_loss_rate
        np.random.normal(-0.32, 0.1),   # system_load
        np.random.normal(0.93, 0.05)    # network_stability
    ])
    
    # 在几个随机维度上添加明显的异常值
    anomaly_dims = np.random.choice(6, size=np.random.randint(1, 3), replace=False)
    for dim in anomaly_dims:
        if dim == 0:  # avg_signal_strength
            anomaly_like_vector[dim] += np.random.uniform(-5, 5)
        elif dim == 1:  # avg_data_rate
            anomaly_like_vector[dim] += np.random.uniform(-0.1, 0.1)
        elif dim == 2:  # avg_latency
            anomaly_like_vector[dim] += np.random.uniform(20, 100)
        elif dim == 3:  # packet_loss_rate
            anomaly_like_vector[dim] += np.random.uniform(0.5, 2.0)
        elif dim == 4:  # system_load
            anomaly_like_vector[dim] += np.random.uniform(-0.5, 1.0)
        elif dim == 5:  # network_stability
            anomaly_like_vector[dim] += np.random.uniform(-0.3, 0.3)
    
    # 随机选择一种类型
    if np.random.random() < 0.3:  # 30%概率生成异常向量
        return anomaly_like_vector, "可能异常"
    else:  # 70%概率生成正常向量
        return normal_like_vector, "可能正常"

def get_default_baseline():
    """获取正常基线数据用于校准"""
    return {
        'wlan0_wireless_quality': 70.0, 'wlan0_wireless_level': -55.0,
        'wlan0_packet_loss_rate': 0.01, 'wlan0_send_rate_bps': 500000.0,
        'wlan0_recv_rate_bps': 1500000.0, 'tcp_retrans_segments': 5,
        'gateway_ping_time': 12.5, 'dns_response_time': 25.0,
        'tcp_connection_count': 30, 'cpu_percent': 15.0, 'memory_percent': 45.0
    }

def test_vector_detection(vector, engine, extractor):
    """
    使用训练好的模型测试向量
    """
    print("=" * 60)
    print("🔍 开始异常检测测试")
    print("=" * 60)
    
    print(f"📊 测试向量（6维）:")
    print(f"   特征值: {vector}")
    print(f"   [信号强度: {vector[0]:.3f}, 数据速率: {vector[1]:.3f}, 延迟: {vector[2]:.3f}")
    print(f"    丢包率: {vector[3]:.3f}, 系统负载: {vector[4]:.3f}, 网络稳定性: {vector[5]:.3f}]")
    print(f"   向量统计: 均值={vector.mean():.3f}, 标准差={vector.std():.3f}")
    print(f"   最大值={vector.max():.3f}, 最小值={vector.min():.3f}")
    print()
    
    try:
        # 获取特征名称
        feature_names = extractor.get_feature_names()
        
        # 进行异常检测
        print("🔬 正在进行异常检测...")
        is_anomaly, details = engine.detect_anomaly_from_vector(vector, feature_names)
        
        # 显示结果
        print(f"   检测结果: {'🚨 异常' if is_anomaly else '✅ 正常'}")
        
        if 'reconstruction_error' in details:
            print(f"   重构误差: {details['reconstruction_error']:.6f}")
        if 'threshold' in details:
            print(f"   检测阈值: {details['threshold']:.6f}")
        print()
        
        if is_anomaly:
            # 显示异常分类信息
            print("🎯 异常分类结果:")
            predicted_class = details.get('predicted_class', 'N/A')
            confidence = details.get('confidence', 0.0)
            print(f"   异常类型: {predicted_class}")
            print(f"   置信度: {confidence:.1%}")
            print()
            
            # 详细分析
            print("📋 详细分析:")
            print(f"   - 这个向量被识别为异常")
            if 'reconstruction_error' in details and 'threshold' in details:
                print(f"   - 重构误差 {details['reconstruction_error']:.6f} 超过了阈值 {details['threshold']:.6f}")
            print(f"   - 异常类型被分类为: {predicted_class}")
            print(f"   - 分类置信度: {confidence:.1%}")
            
            return {
                'is_anomaly': True,
                'anomaly_type': predicted_class,
                'confidence': confidence,
                'details': details
            }
        else:
            print("📋 详细分析:")
            print(f"   - 这个向量被识别为正常")
            if 'reconstruction_error' in details and 'threshold' in details:
                print(f"   - 重构误差 {details['reconstruction_error']:.6f} 在正常范围内（阈值: {details['threshold']:.6f}）")
            print(f"   - 无需进行异常分类")
            
            return {
                'is_anomaly': False,
                'details': details
            }
            
    except Exception as e:
        print(f"❌ 检测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    主测试函数
    """
    print("🚀 AI网络异常检测系统 - 单向量测试")
    print()
    
    try:
        # 初始化系统组件
        print("--- 正在初始化AI引擎和模型，请稍候... ---")
        config = load_config()
        logger = SystemLogger(config['logging'])
        
        # 设置日志级别为WARNING，获得更干净的输出
        logger.set_log_level('WARNING')
        
        extractor = FeatureExtractor(config['data_collection']['metrics'], logger)
        autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
        classifier = ErrorClassifier(config['ai_models']['classifier'], logger)
        
        engine = AnomalyDetectionEngine(
            config=config['anomaly_detection'],
            autoencoder=autoencoder, 
            error_classifier=classifier,
            buffer_manager=None, 
            logger=logger
        )
        
        # 校准特征提取器
        print("--- 正在校准AI模型基准... ---")
        baseline_data = get_default_baseline()
        extractor.extract_features(baseline_data)
        
        print("--- 初始化完成 ---")
        print()
        
        # 生成随机测试向量
        test_vector, expected = generate_random_vector()
        print(f"📝 生成测试向量（预期: {expected}）")
        print()
        
        # 进行检测
        result = test_vector_detection(test_vector, engine, extractor)
        
        if result:
            print("=" * 60)
            print("✨ 测试完成")
            if result['is_anomaly']:
                print(f"🚨 检测结果: 异常 - {result['anomaly_type']} (置信度: {result['confidence']:.1%})")
            else:
                print("✅ 检测结果: 正常")
            print("=" * 60)
        
        # 询问是否继续测试
        print()
        while True:
            choice = input("是否要测试另一个随机向量？(y/n): ").lower().strip()
            if choice == 'y':
                print("\n" + "="*80 + "\n")
                test_vector, expected = generate_random_vector()
                print(f"📝 生成新的测试向量（预期: {expected}）")
                print()
                result = test_vector_detection(test_vector, engine, extractor)
                if result:
                    print("=" * 60)
                    print("✨ 测试完成")
                    if result['is_anomaly']:
                        print(f"🚨 检测结果: 异常 - {result['anomaly_type']} (置信度: {result['confidence']:.1%})")
                    else:
                        print("✅ 检测结果: 正常")
                    print("=" * 60)
                print()
            elif choice == 'n':
                print("👋 测试结束，感谢使用！")
                break
            else:
                print("请输入 'y' 或 'n'")
                
    except KeyboardInterrupt:
        print("\n程序已退出。感谢使用！")
    except Exception as e:
        print(f"\n发生致命错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查配置文件和模型文件是否完整。")

if __name__ == "__main__":
    main() 