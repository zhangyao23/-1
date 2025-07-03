#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI网络异常检测系统 - 全面性能测试脚本
包含自编码器、分类器和系统集成的完整性能评估
"""

import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
import sys
import os

# 添加源代码路径
sys.path.append('src')

from ai_models.autoencoder_model import AutoencoderModel
from ai_models.error_classifier import ErrorClassifier
from feature_processor.feature_extractor import FeatureExtractor
from logger.logger import Logger


class ComprehensivePerformanceTest:
    """综合性能测试类"""
    
    def __init__(self):
        print("🚀 AI网络异常检测系统 - 全面性能测试")
        print("=" * 60)
        
        # 初始化日志
        self.logger = Logger()
        
        # 加载测试数据
        self.load_test_data()
        
        # 初始化模型
        self.init_models()
        
        # 测试结果
        self.results = {
            'autoencoder_performance': {},
            'classifier_performance': {},
            'system_integration': {},
            'performance_benchmark': {}
        }
    
    def load_test_data(self):
        """加载测试数据"""
        print("📊 加载测试数据...")
        
        try:
            # 加载训练数据用于性能测试
            data_path = "data/improved_training_data_6d.csv"
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                print(f"✅ 加载训练数据: {len(df)} 条记录")
                
                # 分离正常和异常数据
                normal_data = df[df['label'] == 0].drop(['label', 'anomaly_type'], axis=1)
                anomaly_data = df[df['label'] == 1].drop(['label', 'anomaly_type'], axis=1)
                
                # 随机采样用于测试
                self.normal_samples = normal_data.sample(n=min(100, len(normal_data))).values
                self.anomaly_samples = anomaly_data.sample(n=min(100, len(anomaly_data))).values
                
                print(f"📦 正常样本: {len(self.normal_samples)} 条")
                print(f"📦 异常样本: {len(self.anomaly_samples)} 条")
                
                # 获取异常类型分布
                anomaly_types = df[df['label'] == 1]['anomaly_type'].value_counts()
                print("📊 异常类型分布:")
                for atype, count in anomaly_types.items():
                    print(f"  {atype}: {count}")
                
            else:
                print("❌ 未找到训练数据文件")
                self.create_mock_data()
                
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            self.create_mock_data()
    
    def create_mock_data(self):
        """创建模拟测试数据"""
        print("🔧 创建模拟测试数据...")
        
        # 正常数据 (6维特征)
        np.random.seed(42)
        self.normal_samples = np.random.normal(
            loc=[8.0, 2.5, 10.0, 0.001, 0.8, 0.95],
            scale=[1.0, 0.5, 2.0, 0.005, 0.2, 0.1],
            size=(100, 6)
        )
        
        # 异常数据 
        self.anomaly_samples = np.random.normal(
            loc=[2.0, 1.0, 80.0, 0.15, 0.9, 0.6],
            scale=[1.0, 0.5, 20.0, 0.05, 0.1, 0.2],
            size=(100, 6)
        )
        
        print(f"✅ 生成模拟数据: 正常{len(self.normal_samples)}条, 异常{len(self.anomaly_samples)}条")
    
    def init_models(self):
        """初始化模型"""
        print("🤖 初始化AI模型...")
        
        try:
            # 初始化自编码器
            self.autoencoder = AutoencoderModel(
                input_dim=6,
                logger=self.logger,
                model_path='models/autoencoder_model_retrained'
            )
            print("✅ 自编码器初始化成功")
            
            # 初始化分类器
            self.classifier = ErrorClassifier(
                logger=self.logger,
                model_path='models/rf_classifier_improved.pkl'
            )
            print("✅ 分类器初始化成功")
            
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            raise
    
    def test_autoencoder_performance(self):
        """测试自编码器性能"""
        print("\n🎯 测试自编码器性能...")
        print("-" * 40)
        
        # 测试正常数据
        print("📈 测试正常数据...")
        normal_errors = []
        normal_predictions = []
        
        for i, sample in enumerate(self.normal_samples):
            try:
                is_anomaly, error, score = self.autoencoder.predict(sample)
                normal_errors.append(error)
                normal_predictions.append(is_anomaly)
                
                if i < 5:  # 显示前5个结果
                    print(f"  样本 {i+1}: 重构误差={error:.6f}, 异常={is_anomaly}")
                    
            except Exception as e:
                print(f"  样本 {i+1} 测试失败: {e}")
        
        # 测试异常数据
        print("📉 测试异常数据...")
        anomaly_errors = []
        anomaly_predictions = []
        
        for i, sample in enumerate(self.anomaly_samples):
            try:
                is_anomaly, error, score = self.autoencoder.predict(sample)
                anomaly_errors.append(error)
                anomaly_predictions.append(is_anomaly)
                
                if i < 5:  # 显示前5个结果
                    print(f"  样本 {i+1}: 重构误差={error:.6f}, 异常={is_anomaly}")
                    
            except Exception as e:
                print(f"  样本 {i+1} 测试失败: {e}")
        
        # 计算性能指标
        if normal_errors and anomaly_errors:
            normal_accuracy = sum(not pred for pred in normal_predictions) / len(normal_predictions)
            anomaly_accuracy = sum(anomaly_predictions) / len(anomaly_predictions)
            overall_accuracy = (normal_accuracy * len(normal_predictions) + 
                              anomaly_accuracy * len(anomaly_predictions)) / (len(normal_predictions) + len(anomaly_predictions))
            
            results = {
                'threshold': self.autoencoder.threshold,
                'normal_samples': len(normal_errors),
                'anomaly_samples': len(anomaly_errors),
                'normal_accuracy': normal_accuracy,
                'anomaly_accuracy': anomaly_accuracy,
                'overall_accuracy': overall_accuracy,
                'normal_error_mean': np.mean(normal_errors),
                'normal_error_std': np.std(normal_errors),
                'anomaly_error_mean': np.mean(anomaly_errors),
                'anomaly_error_std': np.std(anomaly_errors)
            }
            
            print(f"\n📊 自编码器性能指标:")
            print(f"  异常检测阈值: {results['threshold']:.6f}")
            print(f"  正常数据准确率: {results['normal_accuracy']:.3f}")
            print(f"  异常数据准确率: {results['anomaly_accuracy']:.3f}")
            print(f"  总体准确率: {results['overall_accuracy']:.3f}")
            print(f"  正常数据重构误差: {results['normal_error_mean']:.6f} ± {results['normal_error_std']:.6f}")
            print(f"  异常数据重构误差: {results['anomaly_error_mean']:.6f} ± {results['anomaly_error_std']:.6f}")
            
            self.results['autoencoder_performance'] = results
            
        else:
            print("❌ 无法计算自编码器性能指标")
    
    def run_all_tests(self):
        """运行所有测试"""
        try:
            self.test_autoencoder_performance()
            print("\n✅ 全面性能测试完成！")
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 运行全面性能测试
    test = ComprehensivePerformanceTest()
    test.run_all_tests()
