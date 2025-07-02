#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实际数据测试脚本

用于全面测试已训练的AI异常检测系统：
1. 交互式单次测试
2. 批量数据测试
3. 性能基准测试
4. 模拟真实场景测试

使用方法：
- 交互式测试: python scripts/test_with_real_data.py --interactive
- 批量测试: python scripts/test_with_real_data.py --batch
- 性能测试: python scripts/test_with_real_data.py --benchmark
- 场景测试: python scripts/test_with_real_data.py --scenarios
- 完整测试: python scripts/test_with_real_data.py --all
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger.system_logger import SystemLogger
from src.ai_models.autoencoder_model import AutoencoderModel
from src.ai_models.error_classifier import ErrorClassifier
from src.anomaly_detector.anomaly_engine import AnomalyDetectionEngine
from src.feature_processor.feature_extractor import FeatureExtractor

class RealDataTester:
    """实际数据测试器"""
    
    def __init__(self, config_path="config/system_config.json"):
        """初始化测试器"""
        self.config = self._load_config(config_path)
        self.logger = SystemLogger(self.config['logging'])
        self.logger.set_log_level('INFO')
        
        # 初始化组件 - 使用与训练时相同的11个真实网络指标
        real_metrics = [
            'wlan0_wireless_quality', 'wlan0_signal_level', 'wlan0_noise_level',
            'wlan0_rx_packets', 'wlan0_tx_packets', 'wlan0_rx_bytes', 'wlan0_tx_bytes',
            'gateway_ping_time', 'dns_resolution_time', 'memory_usage_percent', 'cpu_usage_percent'
        ]
        self.feature_extractor = FeatureExtractor(
            real_metrics, 
            self.logger,
            scaler_path=os.path.join(self.config['ai_models']['autoencoder']['model_path'], 'autoencoder_scaler.pkl')
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
        
        # 测试统计
        self.test_results = []
        
        print("🚀 AI异常检测系统测试器已初始化")
        
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
    
    def interactive_test(self):
        """交互式单次测试"""
        print("\n" + "="*50)
        print("🔍 交互式AI异常检测测试")
        print("="*50)
        
        # 加载默认数据
        default_data = self._get_default_data()
        
        print("\n📝 请输入网络指标数据（直接回车使用默认值）:")
        input_data = {}
        
        for key, default_value in default_data.items():
            while True:
                try:
                    prompt = f"  {key} (默认: {default_value}): "
                    user_input = input(prompt).strip()
                    
                    if not user_input:
                        input_data[key] = default_value
                        break
                    
                    input_data[key] = float(user_input)
                    break
                except ValueError:
                    print("    ❌ 请输入有效数字")
                except KeyboardInterrupt:
                    print("\n\n👋 测试已取消")
                    return
        
        # 执行检测
        result = self._perform_detection(input_data, "交互式测试")
        self._display_single_result(result)
    
    def batch_test(self):
        """批量数据测试"""
        print("\n" + "="*50)
        print("📊 批量数据测试")
        print("="*50)
        
        # 生成测试数据集
        test_cases = self._generate_test_cases()
        
        print(f"🧪 开始测试 {len(test_cases)} 个测试用例...")
        
        results = []
        for i, (name, data) in enumerate(test_cases.items(), 1):
            print(f"\n[{i}/{len(test_cases)}] 测试: {name}")
            result = self._perform_detection(data, name)
            results.append(result)
            
            # 显示简要结果
            status = "🔴 异常" if result['is_anomaly'] else "🟢 正常"
            if result['is_anomaly']:
                print(f"  结果: {status} ({result['details'].get('predicted_class', 'unknown')})")
            else:
                print(f"  结果: {status}")
        
        # 显示批量测试总结
        self._display_batch_summary(results)
    
    def benchmark_test(self):
        """性能基准测试"""
        print("\n" + "="*50)
        print("⚡ 性能基准测试")
        print("="*50)
        
        # 准备测试数据
        normal_data = self._get_default_data()
        test_sizes = [1, 10, 50, 100, 500]
        
        print("🔥 开始性能基准测试...")
        
        for size in test_sizes:
            print(f"\n测试批量大小: {size}")
            
            # 准备数据
            batch_data = [normal_data.copy() for _ in range(size)]
            
            # 执行性能测试
            start_time = time.time()
            for data in batch_data:
                self._perform_detection(data, f"benchmark_{size}")
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_sample = total_time / size
            throughput = size / total_time
            
            print(f"  总时间: {total_time:.3f}s")
            print(f"  平均每样本: {avg_time_per_sample:.4f}s")
            print(f"  吞吐量: {throughput:.1f} samples/s")
    
    def scenario_test(self):
        """模拟真实场景测试"""
        print("\n" + "="*50)
        print("🎭 真实场景模拟测试")
        print("="*50)
        
        scenarios = self._get_realistic_scenarios()
        
        print(f"🎬 测试 {len(scenarios)} 个真实场景...")
        
        for i, (scenario_name, scenario_data) in enumerate(scenarios.items(), 1):
            print(f"\n[{i}/{len(scenarios)}] 场景: {scenario_name}")
            print(f"  描述: {scenario_data.get('description', '无描述')}")
            
            result = self._perform_detection(scenario_data['data'], scenario_name)
            
            # 显示场景结果
            expected = scenario_data.get('expected_anomaly', None)
            actual = result['is_anomaly']
            
            if expected is not None:
                if expected == actual:
                    print(f"  ✅ 预期: {'异常' if expected else '正常'}, 实际: {'异常' if actual else '正常'}")
                else:
                    print(f"  ❌ 预期: {'异常' if expected else '正常'}, 实际: {'异常' if actual else '正常'}")
            else:
                print(f"  📋 检测结果: {'🔴 异常' if actual else '🟢 正常'}")
            
            if result['is_anomaly']:
                predicted_class = result['details'].get('predicted_class', 'unknown')
                confidence = result['details'].get('confidence', 0.0)
                print(f"  🎯 异常类型: {predicted_class} (置信度: {confidence:.2%})")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("🚀 运行完整测试套件")
        print("="*60)
        
        print("\n1️⃣ 批量数据测试")
        self.batch_test()
        
        print("\n2️⃣ 性能基准测试")
        self.benchmark_test()
        
        print("\n3️⃣ 真实场景测试")
        self.scenario_test()
        
        print("\n" + "="*60)
        print("✅ 完整测试套件执行完毕")
        print("="*60)
    
    def _perform_detection(self, input_data: Dict, test_name: str) -> Dict:
        """执行异常检测"""
        start_time = time.time()
        
        try:
            # 将11个实际网络指标转换为标准格式
            metrics_data = {
                'wlan0_wireless_quality': input_data.get('wlan0_wireless_quality', 70.0),
                'wlan0_signal_level': input_data.get('wlan0_wireless_level', -55.0),
                'wlan0_noise_level': input_data.get('wlan0_noise_level', -95.0),
                'wlan0_rx_packets': input_data.get('wlan0_recv_rate_bps', 1500000.0) / 1000,
                'wlan0_tx_packets': input_data.get('wlan0_send_rate_bps', 500000.0) / 1000,
                'wlan0_rx_bytes': input_data.get('wlan0_recv_rate_bps', 1500000.0),
                'wlan0_tx_bytes': input_data.get('wlan0_send_rate_bps', 500000.0),
                'gateway_ping_time': input_data.get('gateway_ping_time', 12.5),
                'dns_resolution_time': input_data.get('dns_response_time', 25.0),
                'memory_usage_percent': input_data.get('memory_percent', 45.0),
                'cpu_usage_percent': input_data.get('cpu_percent', 15.0)
            }
            
            # 使用FeatureExtractor提取24维特征（与训练时完全一致）
            feature_vector = self.feature_extractor.extract_features(metrics_data)
            
            if feature_vector.size == 0:
                return {
                    'test_name': test_name,
                    'is_anomaly': False,
                    'error': '特征提取失败',
                    'detection_time': time.time() - start_time
                }
            
            # 使用训练时的特征名称（24个特征）
            feature_names = [f'feature_{i}' for i in range(len(feature_vector))]
            
            # 异常检测
            is_anomaly, details = self.engine.detect_anomaly_from_vector(
                feature_vector, feature_names
            )
            
            detection_time = time.time() - start_time
            
            result = {
                'test_name': test_name,
                'is_anomaly': is_anomaly,
                'details': details,
                'detection_time': detection_time,
                'input_data': input_data,
                'feature_vector': feature_vector.tolist()
            }
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            return {
                'test_name': test_name,
                'is_anomaly': False,
                'error': str(e),
                'detection_time': time.time() - start_time
            }
    
    # 删除这个方法，不再需要手工映射特征
    # 现在统一使用FeatureExtractor进行24维特征提取
    
    def _display_single_result(self, result: Dict):
        """显示单个检测结果"""
        print("\n" + "="*40)
        print("🔍 检测结果")
        print("="*40)
        
        if 'error' in result:
            print(f"❌ 错误: {result['error']}")
            return
        
        # 基本信息
        status = "🔴 检测到异常!" if result['is_anomaly'] else "🟢 一切正常"
        print(f"状态: {status}")
        print(f"检测耗时: {result['detection_time']:.4f}s")
        
        if result['is_anomaly']:
            details = result['details']
            predicted_class = details.get('predicted_class', 'unknown')
            confidence = details.get('confidence', 0.0)
            
            print(f"异常类型: {predicted_class}")
            print(f"置信度: {confidence:.2%}")
        
        # 技术详情
        details = result['details']
        reconstruction_error = details.get('reconstruction_error', 'N/A')
        threshold = details.get('threshold', 'N/A')
        
        print(f"\n📊 技术指标:")
        print(f"  重构误差: {reconstruction_error}")
        print(f"  异常阈值: {threshold}")
        
        if isinstance(reconstruction_error, (int, float)) and isinstance(threshold, (int, float)):
            anomaly_score = reconstruction_error / threshold
            print(f"  异常分数: {anomaly_score:.3f}")
    
    def _display_batch_summary(self, results: List[Dict]):
        """显示批量测试总结"""
        print("\n" + "="*40)
        print("📊 批量测试总结")
        print("="*40)
        
        total_tests = len(results)
        anomaly_count = sum(1 for r in results if r.get('is_anomaly', False))
        normal_count = total_tests - anomaly_count
        error_count = sum(1 for r in results if 'error' in r)
        
        avg_time = np.mean([r.get('detection_time', 0) for r in results])
        
        print(f"总测试数: {total_tests}")
        print(f"正常样本: {normal_count} ({normal_count/total_tests*100:.1f}%)")
        print(f"异常样本: {anomaly_count} ({anomaly_count/total_tests*100:.1f}%)")
        print(f"错误数: {error_count}")
        print(f"平均检测时间: {avg_time:.4f}s")
        
        # 异常类型分布
        if anomaly_count > 0:
            anomaly_types = {}
            for r in results:
                if r.get('is_anomaly', False) and 'details' in r:
                    predicted_class = r['details'].get('predicted_class', 'unknown')
                    anomaly_types[predicted_class] = anomaly_types.get(predicted_class, 0) + 1
            
            print(f"\n🎯 异常类型分布:")
            for anomaly_type, count in anomaly_types.items():
                print(f"  {anomaly_type}: {count} ({count/anomaly_count*100:.1f}%)")
    
    def _get_default_data(self) -> Dict:
        """获取默认测试数据"""
        return {
            'wlan0_wireless_quality': 70.0,
            'wlan0_wireless_level': -55.0,
            'wlan0_packet_loss_rate': 0.01,
            'wlan0_send_rate_bps': 500000.0,
            'wlan0_recv_rate_bps': 1500000.0,
            'tcp_retrans_segments': 5,
            'gateway_ping_time': 12.5,
            'dns_response_time': 25.0,
            'tcp_connection_count': 30,
            'cpu_percent': 15.0,
            'memory_percent': 45.0
        }
    
    def _generate_test_cases(self) -> Dict[str, Dict]:
        """生成批量测试用例"""
        base_data = self._get_default_data()
        test_cases = {}
        
        # 正常情况
        test_cases['正常_标准'] = base_data.copy()
        
        # 轻微变化的正常情况
        normal_variant = base_data.copy()
        normal_variant['cpu_percent'] = 20.0
        normal_variant['memory_percent'] = 50.0
        test_cases['正常_轻微变化'] = normal_variant
        
        # 各种异常情况
        # CPU过载
        cpu_overload = base_data.copy()
        cpu_overload['cpu_percent'] = 95.0
        cpu_overload['memory_percent'] = 80.0
        test_cases['CPU过载'] = cpu_overload
        
        # 网络延迟异常
        high_latency = base_data.copy()
        high_latency['gateway_ping_time'] = 500.0
        high_latency['dns_response_time'] = 1000.0
        test_cases['网络延迟异常'] = high_latency
        
        # 数据包丢失异常
        packet_loss = base_data.copy()
        packet_loss['wlan0_packet_loss_rate'] = 0.15
        packet_loss['tcp_retrans_segments'] = 50
        test_cases['数据包丢失异常'] = packet_loss
        
        # 信号质量差
        poor_signal = base_data.copy()
        poor_signal['wlan0_wireless_quality'] = 20.0
        poor_signal['wlan0_wireless_level'] = -85.0
        test_cases['信号质量差'] = poor_signal
        
        # 带宽异常
        bandwidth_issue = base_data.copy()
        bandwidth_issue['wlan0_send_rate_bps'] = 50000.0
        bandwidth_issue['wlan0_recv_rate_bps'] = 100000.0
        test_cases['带宽异常'] = bandwidth_issue
        
        return test_cases
    
    def _get_realistic_scenarios(self) -> Dict[str, Dict]:
        """获取真实场景测试数据"""
        base_data = self._get_default_data()
        scenarios = {}
        
        # 场景1: 视频会议期间网络拥塞
        video_call_congestion = {
            'description': '视频会议期间网络拥塞，延迟增加',
            'expected_anomaly': True,
            'data': {
                **base_data,
                'gateway_ping_time': 150.0,
                'dns_response_time': 200.0,
                'wlan0_packet_loss_rate': 0.08,
                'tcp_connection_count': 45
            }
        }
        scenarios['视频会议网络拥塞'] = video_call_congestion
        
        # 场景2: 深夜正常浏览
        night_browsing = {
            'description': '深夜正常网页浏览，系统负载低',
            'expected_anomaly': False,
            'data': {
                **base_data,
                'cpu_percent': 8.0,
                'memory_percent': 35.0,
                'tcp_connection_count': 15,
                'gateway_ping_time': 8.0
            }
        }
        scenarios['深夜正常浏览'] = night_browsing
        
        # 场景3: 下载大文件时系统异常
        large_download_issue = {
            'description': '下载大文件时出现系统资源异常',
            'expected_anomaly': True,
            'data': {
                **base_data,
                'cpu_percent': 85.0,
                'memory_percent': 90.0,
                'wlan0_recv_rate_bps': 2000000.0,
                'tcp_connection_count': 8
            }
        }
        scenarios['大文件下载异常'] = large_download_issue
        
        # 场景4: 移动设备漫游
        mobile_roaming = {
            'description': '移动设备在不同WiFi间漫游',
            'expected_anomaly': None,  # 不确定是否应该被检测为异常
            'data': {
                **base_data,
                'wlan0_wireless_quality': 45.0,
                'wlan0_wireless_level': -70.0,
                'gateway_ping_time': 35.0,
                'wlan0_packet_loss_rate': 0.03
            }
        }
        scenarios['设备漫游'] = mobile_roaming
        
        return scenarios

    def test_realistic_scenarios(self, scenarios_file: str = "data/realistic_test_scenarios.json"):
        """测试真实错误场景数据"""
        print("\n" + "="*60)
        print("🌐 真实网络错误场景测试")
        print("="*60)
        
        try:
            # 加载场景数据
            with open(scenarios_file, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
            
            print(f"📊 加载了 {len(scenarios)} 个真实场景")
            
            # 按类型分组统计
            results_by_type = {}
            all_results = []
            
            print(f"\n🧪 开始逐个测试场景...")
            
            for i, scenario in enumerate(scenarios, 1):
                scenario_name = scenario['name']
                scenario_type = scenario.get('expected_type', 'unknown')
                expected_anomaly = scenario.get('expected_anomaly', None)
                
                print(f"\n[{i}/{len(scenarios)}] 测试: {scenario_name}")
                print(f"  类型: {scenario_type}")
                print(f"  描述: {scenario['description']}")
                
                # 执行检测
                result = self._perform_detection(scenario['data'], scenario_name)
                result['scenario_type'] = scenario_type
                result['expected_anomaly'] = expected_anomaly
                
                all_results.append(result)
                
                # 按类型分组
                if scenario_type not in results_by_type:
                    results_by_type[scenario_type] = []
                results_by_type[scenario_type].append(result)
                
                # 显示结果
                actual_anomaly = result.get('is_anomaly', False)
                if expected_anomaly is not None:
                    if expected_anomaly == actual_anomaly:
                        status_icon = "✅"
                    else:
                        status_icon = "❌"
                    print(f"  {status_icon} 预期: {'异常' if expected_anomaly else '正常'}, 实际: {'异常' if actual_anomaly else '正常'}")
                else:
                    status_icon = "🔴" if actual_anomaly else "🟢"
                    print(f"  {status_icon} 检测结果: {'异常' if actual_anomaly else '正常'}")
                
                if actual_anomaly:
                    predicted_class = result['details'].get('predicted_class', 'unknown')
                    confidence = result['details'].get('confidence', 0.0)
                    print(f"  🎯 检测类型: {predicted_class} (置信度: {confidence:.1%})")
                    
                    # 检查是否与预期类型匹配
                    if scenario_type in predicted_class or predicted_class in scenario_type:
                        print(f"  🎯 类型匹配: 符合预期")
                    else:
                        print(f"  ⚠️  类型差异: 预期 {scenario_type}, 检测到 {predicted_class}")
            
            # 显示详细分析结果
            self._display_realistic_scenario_analysis(all_results, results_by_type)
            
        except FileNotFoundError:
            print(f"❌ 场景文件未找到: {scenarios_file}")
            print("请先运行 python scripts/generate_realistic_test_data.py 生成测试数据")
        except json.JSONDecodeError:
            print(f"❌ 场景文件格式错误: {scenarios_file}")
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
    
    def _display_realistic_scenario_analysis(self, all_results: List[Dict], results_by_type: Dict):
        """显示真实场景测试分析结果"""
        print("\n" + "="*60)
        print("📈 真实场景测试分析报告")
        print("="*60)
        
        total_scenarios = len(all_results)
        anomaly_count = sum(1 for r in all_results if r.get('is_anomaly', False))
        normal_count = total_scenarios - anomaly_count
        
        # 基本统计
        print(f"\n📊 基本统计:")
        print(f"  总场景数: {total_scenarios}")
        print(f"  检测为异常: {anomaly_count} ({anomaly_count/total_scenarios*100:.1f}%)")
        print(f"  检测为正常: {normal_count} ({normal_count/total_scenarios*100:.1f}%)")
        
        # 预期vs实际对比
        with_expectation = [r for r in all_results if r.get('expected_anomaly') is not None]
        if with_expectation:
            correct_predictions = sum(1 for r in with_expectation 
                                    if r.get('expected_anomaly') == r.get('is_anomaly'))
            accuracy = correct_predictions / len(with_expectation) * 100
            print(f"\n🎯 预测准确性:")
            print(f"  有预期结果的场景: {len(with_expectation)}")
            print(f"  预测正确: {correct_predictions}")
            print(f"  准确率: {accuracy:.1f}%")
        
        # 按错误类型分析
        print(f"\n🔍 错误类型检测分析:")
        for error_type, type_results in sorted(results_by_type.items()):
            anomalies_in_type = sum(1 for r in type_results if r.get('is_anomaly', False))
            detection_rate = anomalies_in_type / len(type_results) * 100
            
            print(f"\n  {error_type} ({len(type_results)} 个场景):")
            print(f"    检测率: {anomalies_in_type}/{len(type_results)} ({detection_rate:.1f}%)")
            
            # 显示检测到的AI模型分类
            detected_classes = {}
            for r in type_results:
                if r.get('is_anomaly', False):
                    predicted_class = r['details'].get('predicted_class', 'unknown')
                    detected_classes[predicted_class] = detected_classes.get(predicted_class, 0) + 1
            
            if detected_classes:
                print(f"    AI分类结果:")
                for ai_class, count in sorted(detected_classes.items()):
                    print(f"      {ai_class}: {count} 次")
        
        # 性能统计
        avg_time = np.mean([r.get('detection_time', 0) for r in all_results])
        print(f"\n⚡ 性能统计:")
        print(f"  平均检测时间: {avg_time:.4f}s")
        print(f"  总检测时间: {sum(r.get('detection_time', 0) for r in all_results):.3f}s")
        
        # 建议和总结
        print(f"\n💡 分析建议:")
        
        # 检测敏感性分析
        if anomaly_count / total_scenarios > 0.8:
            print("  📈 系统检测敏感性较高，倾向于安全优先策略")
        elif anomaly_count / total_scenarios < 0.3:
            print("  📉 系统检测敏感性较低，可能存在漏检风险")
        else:
            print("  ⚖️  系统检测敏感性适中")
        
        # 类型匹配分析
        normal_scenarios = [r for r in all_results if r.get('scenario_type', '').startswith('normal')]
        if normal_scenarios:
            false_positive_rate = sum(1 for r in normal_scenarios if r.get('is_anomaly', False)) / len(normal_scenarios)
            if false_positive_rate > 0.5:
                print("  ⚠️  正常场景误报率较高，建议调整检测阈值")
        
        print(f"\n🎯 建议下一步:")
        print(f"  1. 根据实际需求调整异常检测阈值")
        print(f"  2. 针对特定错误类型优化特征映射")
        print(f"  3. 收集更多真实场景数据进行模型微调")
        print(f"  4. 考虑为不同应用环境设置不同的检测参数")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI异常检测系统实际数据测试工具")
    parser.add_argument('--interactive', action='store_true', help='运行交互式单次测试')
    parser.add_argument('--batch', action='store_true', help='运行批量数据测试')
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')
    parser.add_argument('--scenarios', action='store_true', help='运行真实场景测试')
    parser.add_argument('--realistic', action='store_true', help='运行真实错误场景测试')
    parser.add_argument('--file', type=str, default='data/realistic_test_scenarios.json', 
                       help='真实场景数据文件路径')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    
    args = parser.parse_args()
    
    try:
        tester = RealDataTester()
        
        if args.interactive:
            tester.interactive_test()
        elif args.batch:
            tester.batch_test()
        elif args.benchmark:
            tester.benchmark_test()
        elif args.scenarios:
            tester.scenario_test()
        elif args.realistic:
            tester.test_realistic_scenarios(args.file)
        elif args.all:
            tester.run_all_tests()
            # 也运行真实场景测试
            print("\n4️⃣ 真实错误场景测试")
            tester.test_realistic_scenarios(args.file)
        else:
            # 默认运行交互式测试
            print("未指定测试模式，运行交互式测试...")
            print("使用 --help 查看所有可用选项")
            tester.interactive_test()
            
    except KeyboardInterrupt:
        print("\n\n👋 测试已取消")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 