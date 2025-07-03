#!/usr/bin/env python3
"""
AI异常检测系统 - 综合功能测试脚本
===================================
系统性地测试所有组件功能，确保系统正常运行
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# 忽略警告信息
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

def print_test_header(test_name):
    """打印测试头部信息"""
    print(f"\n{'='*60}")
    print(f"🔍 测试: {test_name}")
    print(f"{'='*60}")

def print_result(test_name, success, message=""):
    """打印测试结果"""
    status = "✅ 通过" if success else "❌ 失败"
    print(f"{status} {test_name}")
    if message:
        print(f"   详情: {message}")

class SystemTester:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        
    def test_project_structure(self):
        """测试项目结构完整性"""
        print_test_header("项目结构检查")
        
        required_dirs = [
            'src', 'models', 'data', 'config', 'scripts', 'test'
        ]
        
        required_files = [
            'requirements.txt', 'README.md',
            'config/system_config.json',
            'models/error_classifier.pkl'
        ]
        
        # 检查目录
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            success = dir_path.exists() and dir_path.is_dir()
            print_result(f"目录 {dir_name}/", success)
            self.test_results[f"dir_{dir_name}"] = success
        
        # 检查文件
        for file_path in required_files:
            full_path = self.project_root / file_path
            success = full_path.exists() and full_path.is_file()
            print_result(f"文件 {file_path}", success)
            self.test_results[f"file_{file_path.replace('/', '_')}"] = success

    def test_config_files(self):
        """测试配置文件"""
        print_test_header("配置文件检查")
        
        # 测试系统配置
        try:
            config_path = self.project_root / 'config' / 'system_config.json'
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            required_keys = ['autoencoder', 'classifier', 'anomaly_types']
            success = all(key in config for key in required_keys)
            print_result("系统配置文件格式", success)
            
            if success:
                print(f"   异常类型数量: {len(config['anomaly_types'])}")
                print(f"   自编码器特征维度: {config['autoencoder']['feature_dim']}")
            
            self.test_results['system_config'] = success
            
        except Exception as e:
            print_result("系统配置文件", False, str(e))
            self.test_results['system_config'] = False

    def test_data_files(self):
        """测试数据文件"""
        print_test_header("数据文件检查")
        
        data_files = {
            '6维正常数据': 'data/6d_normal_traffic.csv',
            '6维异常数据': 'data/6d_labeled_anomalies.csv'
        }
        
        for name, file_path in data_files.items():
            try:
                full_path = self.project_root / file_path
                if full_path.exists():
                    df = pd.read_csv(full_path)
                    print_result(f"{name}", True, f"样本数: {len(df)}, 特征数: {df.shape[1]}")
                    self.test_results[f"data_{name}"] = True
                else:
                    print_result(f"{name}", False, "文件不存在")
                    self.test_results[f"data_{name}"] = False
                    
            except Exception as e:
                print_result(f"{name}", False, str(e))
                self.test_results[f"data_{name}"] = False

    def test_model_files(self):
        """测试模型文件"""
        print_test_header("模型文件检查")
        
        # 测试分类器
        try:
            classifier_path = self.project_root / 'models' / 'error_classifier.pkl'
            with open(classifier_path, 'rb') as f:
                classifier = pickle.load(f)
            
            print_result("随机森林分类器", True, f"类型: {type(classifier).__name__}")
            self.test_results['classifier_model'] = True
            
        except Exception as e:
            print_result("随机森林分类器", False, str(e))
            self.test_results['classifier_model'] = False
        
        # 测试自编码器
        try:
            autoencoder_path = self.project_root / 'models' / 'autoencoder_model'
            success = autoencoder_path.exists() and (autoencoder_path / 'saved_model.pb').exists()
            print_result("自编码器模型", success)
            self.test_results['autoencoder_model'] = success
            
        except Exception as e:
            print_result("自编码器模型", False, str(e))
            self.test_results['autoencoder_model'] = False

    def test_core_modules(self):
        """测试核心模块导入"""
        print_test_header("核心模块导入测试")
        
        modules_to_test = [
            ('特征处理器', 'src.feature_processor.feature_extractor'),
            ('异常检测器', 'src.anomaly_detector.detector'),
            ('AI模型', 'src.ai_models.autoencoder'),
            ('主程序', 'src.main')
        ]
        
        for name, module_path in modules_to_test:
            try:
                __import__(module_path)
                print_result(f"{name}模块", True)
                self.test_results[f"module_{name}"] = True
            except Exception as e:
                print_result(f"{name}模块", False, str(e))
                self.test_results[f"module_{name}"] = False

    def test_feature_extraction(self):
        """测试特征提取功能"""
        print_test_header("特征提取功能测试")
        
        try:
            from src.feature_processor.feature_extractor import FeatureExtractor
            
            extractor = FeatureExtractor()
            
            # 构造测试数据
            test_data = {
                'wlan0_wireless_quality': 75.0,
                'wlan0_send_rate': 800000.0,
                'wlan0_recv_rate': 1200000.0,
                'ping_time': 25.0,
                'dns_resolve_time': 15.0,
                'packet_loss': 0.02,
                'retransmissions': 5.0,
                'cpu_percent': 20.0,
                'memory_percent': 45.0,
                'disk_io_read': 1000000,
                'disk_io_write': 500000
            }
            
            # 提取特征
            features = extractor._convert_to_vector(test_data)
            
            success = (
                isinstance(features, np.ndarray) and 
                features.shape == (6,) and
                not np.isnan(features).any()
            )
            
            print_result("特征提取", success, f"输出维度: {features.shape}")
            if success:
                print(f"   特征值: {features}")
                
            self.test_results['feature_extraction'] = success
            
        except Exception as e:
            print_result("特征提取", False, str(e))
            self.test_results['feature_extraction'] = False

    def test_anomaly_detection(self):
        """测试异常检测功能"""
        print_test_header("异常检测功能测试")
        
        try:
            # 导入异常检测器
            from src.anomaly_detector.detector import AnomalyDetector
            
            detector = AnomalyDetector()
            
            # 测试正常数据
            normal_data = {
                'wlan0_wireless_quality': 80.0,
                'wlan0_send_rate': 1000000.0,
                'wlan0_recv_rate': 1500000.0,
                'ping_time': 20.0,
                'dns_resolve_time': 10.0,
                'packet_loss': 0.01,
                'retransmissions': 2.0,
                'cpu_percent': 15.0,
                'memory_percent': 40.0,
                'disk_io_read': 800000,
                'disk_io_write': 400000
            }
            
            result = detector.detect_anomaly(normal_data)
            
            success = (
                isinstance(result, dict) and
                'is_anomaly' in result and
                'reconstruction_error' in result
            )
            
            print_result("正常数据检测", success)
            if success:
                print(f"   检测结果: {'异常' if result['is_anomaly'] else '正常'}")
                print(f"   重构误差: {result['reconstruction_error']:.4f}")
            
            # 测试异常数据
            anomaly_data = {
                'wlan0_wireless_quality': 30.0,  # 信号很差
                'wlan0_send_rate': 100000.0,     # 传输速率很低
                'wlan0_recv_rate': 200000.0,
                'ping_time': 150.0,              # 延迟很高
                'dns_resolve_time': 80.0,
                'packet_loss': 0.15,             # 丢包率高
                'retransmissions': 50.0,
                'cpu_percent': 85.0,             # CPU使用率高
                'memory_percent': 90.0,          # 内存使用率高
                'disk_io_read': 100000,
                'disk_io_write': 50000
            }
            
            anomaly_result = detector.detect_anomaly(anomaly_data)
            
            anomaly_success = (
                isinstance(anomaly_result, dict) and
                'is_anomaly' in anomaly_result
            )
            
            print_result("异常数据检测", anomaly_success)
            if anomaly_success:
                print(f"   检测结果: {'异常' if anomaly_result['is_anomaly'] else '正常'}")
                print(f"   重构误差: {anomaly_result['reconstruction_error']:.4f}")
                if anomaly_result['is_anomaly'] and 'anomaly_type' in anomaly_result:
                    print(f"   异常类型: {anomaly_result['anomaly_type']}")
            
            self.test_results['anomaly_detection'] = success and anomaly_success
            
        except Exception as e:
            print_result("异常检测", False, str(e))
            self.test_results['anomaly_detection'] = False

    def test_scripts(self):
        """测试脚本文件语法"""
        print_test_header("脚本文件语法检查")
        
        script_files = [
            'scripts/train_model.py',
            'scripts/interactive_tester.py',
            'scripts/test_scenarios.py',
            'scripts/generate_simple_6d_data.py'
        ]
        
        for script_file in script_files:
            try:
                script_path = self.project_root / script_file
                if script_path.exists():
                    # 检查语法
                    with open(script_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    compile(code, script_file, 'exec')
                    print_result(f"脚本 {script_file}", True, "语法正确")
                else:
                    print_result(f"脚本 {script_file}", False, "文件不存在")
                    
            except SyntaxError as e:
                print_result(f"脚本 {script_file}", False, f"语法错误: {e}")
            except Exception as e:
                print_result(f"脚本 {script_file}", False, str(e))

    def test_data_generation(self):
        """测试数据生成功能"""
        print_test_header("数据生成功能测试")
        
        try:
            # 测试简单6维数据生成
            exec_path = self.project_root / 'scripts' / 'generate_simple_6d_data.py'
            if exec_path.exists():
                # 导入并测试数据生成函数
                spec = __import__('importlib.util').util.spec_from_file_location(
                    "data_generator", exec_path
                )
                data_gen_module = __import__('importlib.util').util.module_from_spec(spec)
                spec.loader.exec_module(data_gen_module)
                
                print_result("数据生成脚本加载", True)
                self.test_results['data_generation'] = True
            else:
                print_result("数据生成脚本", False, "文件不存在")
                self.test_results['data_generation'] = False
                
        except Exception as e:
            print_result("数据生成功能", False, str(e))
            self.test_results['data_generation'] = False

    def generate_test_report(self):
        """生成测试报告"""
        print_test_header("测试报告总结")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"📊 测试统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过数量: {passed_tests}")
        print(f"   失败数量: {failed_tests}")
        print(f"   通过率: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ 失败的测试:")
            for test_name, result in self.test_results.items():
                if not result:
                    print(f"   - {test_name}")
        
        # 保存测试报告
        report_path = self.project_root / 'test' / 'test_report.json'
        report_data = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': passed_tests/total_tests*100,
            'detailed_results': self.test_results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 详细测试报告已保存到: {report_path}")
        
        return passed_tests == total_tests

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始AI异常检测系统综合测试...")
        
        # 执行所有测试
        self.test_project_structure()
        self.test_config_files()
        self.test_data_files()
        self.test_model_files()
        self.test_core_modules()
        self.test_feature_extraction()
        self.test_anomaly_detection()
        self.test_scripts()
        self.test_data_generation()
        
        # 生成报告
        all_passed = self.generate_test_report()
        
        if all_passed:
            print("\n🎉 所有测试通过！系统功能正常。")
        else:
            print("\n⚠️  部分测试失败，请检查上述错误信息。")
        
        return all_passed

def main():
    """主函数"""
    tester = SystemTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 