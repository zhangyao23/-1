#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
阈值调优脚本 - 优化异常检测系统的检测阈值
帮助找到最佳的异常检测阈值，平衡误报率和检测率
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import argparse
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger.system_logger import SystemLogger
from src.ai_models.autoencoder_model import AutoencoderModel
from src.ai_models.error_classifier import ErrorClassifier
from src.anomaly_detector.anomaly_engine import AnomalyDetectionEngine
from src.feature_processor.feature_extractor import FeatureExtractor


class ThresholdTuner:
    """异常检测阈值调优器"""
    
    def __init__(self, config_path="config/system_config.json"):
        """初始化调优器"""
        self.config = self._load_config(config_path)
        self.logger = SystemLogger(self.config['logging'])
        self.logger.set_log_level('INFO')
        
        # 初始化组件
        self.feature_extractor = FeatureExtractor(
            self.config['data_collection']['metrics'], 
            self.logger
        )
        
        self.autoencoder = AutoencoderModel(
            self.config['ai_models']['autoencoder'], 
            self.logger
        )
        
        self.classifier = ErrorClassifier(
            self.config['ai_models']['classifier'], 
            self.logger
        )
        
        self.engine = AnomalyDetectionEngine(
            config=self.config['anomaly_detection'],
            autoencoder=self.autoencoder,
            error_classifier=self.classifier,
            buffer_manager=None,
            logger=self.logger
        )
        
        self.original_threshold = self.autoencoder.threshold
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ 配置文件 '{config_path}' 未找到")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ 配置文件 '{config_path}' 格式错误")
            sys.exit(1)
        
    def initialize_detector(self) -> bool:
        """初始化异常检测器（用于兼容性）"""
        print(f"🔧 异常检测器初始化完成")
        print(f"📊 当前阈值: {self.original_threshold:.6f}")
        return True
    
    def load_test_scenarios(self, scenarios_file: str) -> List[Dict]:
        """加载测试场景数据"""
        try:
            with open(scenarios_file, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
            print(f"📁 已加载 {len(scenarios)} 个测试场景")
            return scenarios
        except Exception as e:
            print(f"❌ 加载测试场景失败: {e}")
            return []
    
    def _map_to_training_features(self, input_data: Dict) -> np.ndarray:
        """将实际数据映射到训练特征格式"""
        # 从实际的11个网络指标映射到28个训练特征
        feature_mapping = {
            # 基础网络指标
            'wlan0_wireless_quality': 0,
            'wlan0_signal_level': 1,
            'wlan0_noise_level': 2,
            'wlan0_rx_packets': 3,
            'wlan0_tx_packets': 4,
            'wlan0_rx_bytes': 5,
            'wlan0_tx_bytes': 6,
            'gateway_ping_time': 7,
            'dns_resolution_time': 8,
            'memory_usage_percent': 9,
            'cpu_usage_percent': 10
        }
        
        # 初始化28个特征的数组
        features = np.zeros(28)
        
        # 映射已知的特征
        for key, value in input_data.items():
            if key in feature_mapping:
                features[feature_mapping[key]] = value
        
        # 生成派生特征
        if 'wlan0_wireless_quality' in input_data and 'wlan0_signal_level' in input_data:
            features[11] = input_data['wlan0_wireless_quality'] / abs(input_data['wlan0_signal_level']) if input_data['wlan0_signal_level'] != 0 else 0
        
        if 'wlan0_rx_packets' in input_data and 'wlan0_tx_packets' in input_data:
            features[12] = (input_data['wlan0_rx_packets'] + input_data['wlan0_tx_packets']) / 2
        
        if 'gateway_ping_time' in input_data and 'dns_resolution_time' in input_data:
            features[13] = (input_data['gateway_ping_time'] + input_data['dns_resolution_time']) / 2
        
        if 'memory_usage_percent' in input_data and 'cpu_usage_percent' in input_data:
            features[14] = (input_data['memory_usage_percent'] + input_data['cpu_usage_percent']) / 2
        
        # 网络性能组合特征
        if 'wlan0_rx_bytes' in input_data and 'wlan0_tx_bytes' in input_data:
            features[15] = input_data['wlan0_rx_bytes'] / (input_data['wlan0_tx_bytes'] + 1)
            features[16] = (input_data['wlan0_rx_bytes'] + input_data['wlan0_tx_bytes']) / 1024
        
        # 延迟相关特征
        if 'gateway_ping_time' in input_data:
            features[17] = np.log1p(input_data['gateway_ping_time'])
            features[18] = input_data['gateway_ping_time'] ** 2
        
        if 'dns_resolution_time' in input_data:
            features[19] = np.log1p(input_data['dns_resolution_time'])
            features[20] = input_data['dns_resolution_time'] ** 2
        
        # 资源利用率特征
        if 'memory_usage_percent' in input_data:
            features[21] = np.log1p(input_data['memory_usage_percent'])
            features[22] = input_data['memory_usage_percent'] ** 2
        
        if 'cpu_usage_percent' in input_data:
            features[23] = np.log1p(input_data['cpu_usage_percent'])
            features[24] = input_data['cpu_usage_percent'] ** 2
        
        # 交互特征
        if 'wlan0_wireless_quality' in input_data and 'gateway_ping_time' in input_data:
            features[25] = input_data['wlan0_wireless_quality'] * input_data['gateway_ping_time']
        
        if 'memory_usage_percent' in input_data and 'cpu_usage_percent' in input_data:
            features[26] = input_data['memory_usage_percent'] * input_data['cpu_usage_percent']
        
        # 综合指标
        signal_quality = input_data.get('wlan0_wireless_quality', 0)
        ping_time = input_data.get('gateway_ping_time', 0)
        cpu_usage = input_data.get('cpu_usage_percent', 0)
        memory_usage = input_data.get('memory_usage_percent', 0)
        
        features[27] = (signal_quality * 0.3 - ping_time * 0.3 - cpu_usage * 0.2 - memory_usage * 0.2)
        
        return features
    
    def evaluate_threshold(self, threshold: float, scenarios: List[Dict]) -> Dict[str, Any]:
        """评估特定阈值下的检测性能"""
        
        # 临时设置新阈值
        original_threshold = self.autoencoder.threshold
        self.autoencoder.threshold = threshold
        
        results = {
            'threshold': threshold,
            'total_scenarios': len(scenarios),
            'detected_anomalies': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'scenario_results': []
        }
        
        for scenario in scenarios:
            scenario_type = scenario.get('expected_type', 'unknown')
            expected_anomaly = scenario.get('expected_anomaly', None)
            
            # 提取特征并检测
            features = self._map_to_training_features(scenario['data'])
            
            # 使用AutoencoderModel的predict方法
            prediction = self.autoencoder.predict(features)
            is_anomaly = prediction.get('is_anomaly', False)
            reconstruction_error = prediction.get('reconstruction_error', 0.0)
            
            results['scenario_results'].append({
                'name': scenario['name'],
                'type': scenario_type,
                'expected_anomaly': expected_anomaly,
                'reconstruction_error': reconstruction_error,
                'detected_anomaly': is_anomaly
            })
            
            if is_anomaly:
                results['detected_anomalies'] += 1
            
            # 计算混淆矩阵
            if expected_anomaly is not None:
                if expected_anomaly and is_anomaly:
                    results['true_positives'] += 1
                elif expected_anomaly and not is_anomaly:
                    results['false_negatives'] += 1
                elif not expected_anomaly and is_anomaly:
                    results['false_positives'] += 1
                elif not expected_anomaly and not is_anomaly:
                    results['true_negatives'] += 1
        
        # 计算性能指标
        tp = results['true_positives']
        fp = results['false_positives']
        tn = results['true_negatives']
        fn = results['false_negatives']
        
        results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        results['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        results['f1_score'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0
        results['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # 恢复原始阈值
        self.autoencoder.threshold = original_threshold
        
        return results
    
    def tune_threshold(self, scenarios: List[Dict], threshold_range: Optional[Tuple[float, float]] = None, num_points: int = 50) -> List[Dict]:
        """调优阈值，找到最佳性能点"""
        print("\n🔍 开始阈值调优...")
        
        if threshold_range is None:
            # 基于重构误差范围自动确定阈值范围
            errors = []
            for scenario in scenarios:
                features = self._map_to_training_features(scenario['data'])
                prediction = self.autoencoder.predict(features)
                error = prediction.get('reconstruction_error', 0.0)
                errors.append(error)
            
            min_error = min(errors)
            max_error = max(errors)
            threshold_range = (min_error * 0.8, max_error * 1.2)
            print(f"📊 自动确定阈值范围: {threshold_range[0]:.6f} - {threshold_range[1]:.6f}")
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        results = []
        
        print(f"🧪 测试 {num_points} 个阈值点...")
        for i, threshold in enumerate(thresholds, 1):
            if i % 10 == 0 or i == 1:
                print(f"  进度: {i}/{num_points} - 当前阈值: {threshold:.6f}")
            
            result = self.evaluate_threshold(threshold, scenarios)
            results.append(result)
        
        return results
    
    def find_optimal_threshold(self, results: List[Dict], optimize_for: str = 'f1') -> Optional[Dict]:
        """找到最佳阈值"""
        print(f"\n🎯 寻找最佳阈值 (优化指标: {optimize_for})...")
        
        if not results:
            return None
            
        best_result = None
        best_score = -1
        
        for result in results:
            if optimize_for == 'f1':
                score = result['f1_score']
            elif optimize_for == 'accuracy':
                score = result['accuracy']
            elif optimize_for == 'balanced':
                # 平衡精确率和召回率
                score = (result['precision'] + result['recall']) / 2
            elif optimize_for == 'low_fpr':
                # 最小化误报率
                score = 1 - result['false_positive_rate']
            else:
                score = result['f1_score']
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def generate_simple_report(self, results: List[Dict], optimal_result: Dict):
        """生成简化的调优报告"""
        print("\n" + "="*60)
        print("🔧 异常检测系统阈值调优报告")
        print("="*60)
        
        # 基本信息
        print("\n📊 调优基本信息:")
        print(f"  原始阈值: {self.original_threshold:.6f}")
        print(f"  测试场景数: {results[0]['total_scenarios']}")
        print(f"  测试阈值数: {len(results)}")
        
        # 最佳阈值结果
        print("\n🎯 推荐阈值:")
        print(f"  最佳阈值: {optimal_result['threshold']:.6f}")
        print(f"  准确率: {optimal_result['accuracy']:.3f}")
        print(f"  精确率: {optimal_result['precision']:.3f}")
        print(f"  召回率: {optimal_result['recall']:.3f}")
        print(f"  F1分数: {optimal_result['f1_score']:.3f}")
        print(f"  误报率: {optimal_result['false_positive_rate']:.3f}")
        
        # 混淆矩阵
        print("\n📈 混淆矩阵:")
        print(f"  真正例 (TP): {optimal_result['true_positives']}")
        print(f"  假正例 (FP): {optimal_result['false_positives']}")
        print(f"  真负例 (TN): {optimal_result['true_negatives']}")
        print(f"  假负例 (FN): {optimal_result['false_negatives']}")
        
        # 各类场景检测情况
        print("\n🔍 各类场景检测情况:")
        scenario_types = {}
        for scenario_result in optimal_result['scenario_results']:
            scenario_type = scenario_result['type']
            if scenario_type not in scenario_types:
                scenario_types[scenario_type] = {'total': 0, 'detected': 0}
            scenario_types[scenario_type]['total'] += 1
            if scenario_result['detected_anomaly']:
                scenario_types[scenario_type]['detected'] += 1
        
        for scenario_type, counts in sorted(scenario_types.items()):
            detection_rate = counts['detected'] / counts['total'] * 100
            print(f"  {scenario_type}: {counts['detected']}/{counts['total']} ({detection_rate:.1f}%)")
        
        # 建议
        print("\n💡 优化建议:")
        if optimal_result['false_positive_rate'] > 0.2:
            print("  ⚠️  误报率较高，建议进一步提高阈值")
        if optimal_result['recall'] < 0.8:
            print("  ⚠️  召回率较低，可能存在漏检风险")
        if optimal_result['f1_score'] > 0.8:
            print("  ✅ F1分数较高，系统性能良好")
        else:
            print("  📈 建议收集更多训练数据优化模型")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="异常检测系统阈值调优工具")
    parser.add_argument('--scenarios', type=str, default='data/realistic_test_scenarios.json',
                       help='测试场景数据文件路径')
    parser.add_argument('--optimize', type=str, choices=['f1', 'accuracy', 'balanced', 'low_fpr'],
                       default='balanced', help='优化目标指标')
    parser.add_argument('--points', type=int, default=30, help='测试阈值点数量')
    
    args = parser.parse_args()
    
    try:
        tuner = ThresholdTuner()
        
        # 初始化检测器
        if not tuner.initialize_detector():
            return
        
        # 加载测试场景
        scenarios = tuner.load_test_scenarios(args.scenarios)
        if not scenarios:
            return
        
        # 执行阈值调优
        results = tuner.tune_threshold(scenarios, None, args.points)
        
        # 找到最佳阈值
        optimal_result = tuner.find_optimal_threshold(results, args.optimize)
        
        if optimal_result:
            # 生成报告
            tuner.generate_simple_report(results, optimal_result)
            
            print(f"\n✅ 阈值调优完成!")
            print(f"📊 推荐阈值: {optimal_result['threshold']:.6f}")
            print(f"📈 F1分数: {optimal_result['f1_score']:.3f}")
            
            # 提供配置更新建议
            print(f"\n🔧 配置更新建议:")
            print(f"可以考虑将系统阈值从 {tuner.original_threshold:.6f} 调整为 {optimal_result['threshold']:.6f}")
        else:
            print("❌ 未能找到最佳阈值")
        
    except KeyboardInterrupt:
        print("\n\n👋 调优已取消")
    except Exception as e:
        print(f"\n❌ 调优过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 