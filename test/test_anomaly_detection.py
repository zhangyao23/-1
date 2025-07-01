#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI异常检测系统测试脚本

测试系统的各个核心模块，验证功能正确性和性能表现。
包括数据采集、特征提取、AI模型、缓冲区管理等测试。

运行方式：
python test/test_anomaly_detection.py
"""

import unittest
import sys
import os
import numpy as np
import tempfile
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger.system_logger import SystemLogger
from src.data_collector.network_collector import NetworkDataCollector
from src.feature_processor.feature_extractor import FeatureExtractor
from src.buffer_manager.circular_buffer import CircularBuffer
from src.ai_models.autoencoder_model import AutoencoderModel
from src.ai_models.error_classifier import ErrorClassifier
from src.anomaly_detector.anomaly_engine import AnomalyDetectionEngine


class TestSystemLogger(unittest.TestCase):
    """测试系统日志模块"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_config = {
            'level': 'DEBUG',
            'file_path': os.path.join(self.temp_dir, 'test.log'),
            'max_file_size': '1MB',
            'backup_count': 3,
            'console_output': False
        }
    
    def test_logger_initialization(self):
        """测试日志系统初始化"""
        logger = SystemLogger(self.log_config)
        self.assertIsNotNone(logger)
        
        # 测试各种级别的日志记录
        logger.debug("调试信息")
        logger.info("信息")
        logger.warning("警告")
        logger.error("错误")
        
        # 检查日志文件是否创建
        self.assertTrue(os.path.exists(self.log_config['file_path']))
    
    def test_log_file_rotation(self):
        """测试日志文件轮转"""
        logger = SystemLogger(self.log_config)
        
        # 写入大量日志触发轮转
        for i in range(1000):
            logger.info(f"测试日志消息 {i} " + "x" * 1000)
        
        # 检查是否有备份文件
        log_dir = Path(self.log_config['file_path']).parent
        log_files = list(log_dir.glob('*.log*'))
        self.assertGreater(len(log_files), 1)


class TestNetworkDataCollector(unittest.TestCase):
    """测试网络数据采集模块"""
    
    def setUp(self):
        self.logger = SystemLogger({
            'level': 'INFO',
            'file_path': '/tmp/test_collector.log',
            'console_output': False
        })
        
        self.config = {
            'interfaces': ['lo'],  # 使用回环接口进行测试
            'collection_interval': 1,
            'metrics': ['throughput', 'connection_count', 'latency'],
            'timeout': 5
        }
    
    def test_data_collection(self):
        """测试数据采集功能"""
        collector = NetworkDataCollector(self.config, self.logger)
        
        # 采集数据
        data = collector.collect_network_data()
        
        self.assertIsInstance(data, dict)
        self.assertIn('timestamp', data)
        self.assertIn('collection_time', data)
    
    def test_interface_validation(self):
        """测试网络接口验证"""
        collector = NetworkDataCollector(self.config, self.logger)
        
        # 回环接口应该总是可用的
        self.assertTrue(collector.validate_interfaces())


class TestFeatureExtractor(unittest.TestCase):
    """测试特征提取模块"""
    
    def setUp(self):
        self.logger = SystemLogger({
            'level': 'INFO',
            'file_path': '/tmp/test_features.log',
            'console_output': False
        })
        
        self.metrics = ['signal_strength', 'packet_loss_rate', 'data_rate']
    
    def test_feature_extraction(self):
        """测试特征提取"""
        extractor = FeatureExtractor(self.metrics, self.logger)
        
        # 模拟网络数据
        raw_data = {
            'wlan0_signal_strength': -45.0,
            'wlan0_packet_loss_rate': 0.1,
            'eth0_send_rate_bps': 1024000,
            'cpu_percent': 25.5,
            'memory_percent': 60.0
        }
        
        features = extractor.extract_features(raw_data)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
    
    def test_data_cleaning(self):
        """测试数据清洗功能"""
        extractor = FeatureExtractor(self.metrics, self.logger)
        
        # 包含异常值的数据
        raw_data = {
            'valid_metric': 10.0,
            'nan_metric': float('nan'),
            'inf_metric': float('inf'),
            'string_metric': 'invalid',
            'bool_metric': True
        }
        
        features = extractor.extract_features(raw_data)
        
        # 应该能够处理异常值而不抛出异常
        self.assertIsInstance(features, np.ndarray)


class TestCircularBuffer(unittest.TestCase):
    """测试环形缓冲区模块"""
    
    def setUp(self):
        self.logger = SystemLogger({
            'level': 'INFO',
            'file_path': '/tmp/test_buffer.log',
            'console_output': False
        })
        
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'buffer_size': 10,
            'data_retention_minutes': 5,
            'save_threshold': 0.8,
            'compression_enabled': False,
            'anomaly_data_path': self.temp_dir
        }
    
    def test_buffer_operations(self):
        """测试缓冲区基本操作"""
        buffer = CircularBuffer(self.config, self.logger)
        
        # 添加数据
        for i in range(5):
            data = {'value': i, 'timestamp': f'2023-01-01T00:0{i}:00'}
            self.assertTrue(buffer.add_data(data))
        
        # 检查大小
        self.assertEqual(len(buffer), 5)
        
        # 获取最近数据
        recent_data = buffer.get_recent_data(3)
        self.assertEqual(len(recent_data), 3)
        
        # 获取所有数据
        all_data = buffer.get_all_data()
        self.assertEqual(len(all_data), 5)
    
    def test_buffer_overflow(self):
        """测试缓冲区溢出处理"""
        buffer = CircularBuffer(self.config, self.logger)
        
        # 添加超过缓冲区大小的数据
        for i in range(15):
            data = {'value': i}
            buffer.add_data(data)
        
        # 缓冲区大小应该限制在配置的最大值
        self.assertEqual(len(buffer), self.config['buffer_size'])
    
    def test_anomaly_data_saving(self):
        """测试异常数据保存"""
        buffer = CircularBuffer(self.config, self.logger)
        
        # 添加一些数据
        for i in range(5):
            buffer.add_data({'value': i})
        
        # 保存异常数据
        anomaly_info = {
            'anomaly_type': 'test_anomaly',
            'confidence': 0.85,
            'timestamp': '2023-01-01T00:00:00'
        }
        
        filepath = buffer.save_anomaly_data(anomaly_info)
        self.assertIsNotNone(filepath)
        self.assertTrue(os.path.exists(filepath))


class TestAutoencoder(unittest.TestCase):
    """测试自编码器模型"""
    
    def setUp(self):
        self.logger = SystemLogger({
            'level': 'INFO',
            'file_path': '/tmp/test_autoencoder.log',
            'console_output': False
        })
        
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'input_features': 10,
            'encoding_dim': 5,
            'threshold': 0.02,
            'batch_size': 32,
            'model_path': os.path.join(self.temp_dir, 'test_autoencoder.h5')
        }
    
    def test_model_building(self):
        """测试模型构建"""
        autoencoder = AutoencoderModel(self.config, self.logger)
        model = autoencoder.build_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape[1], self.config['input_features'])
    
    def test_training_and_prediction(self):
        """测试模型训练和预测"""
        autoencoder = AutoencoderModel(self.config, self.logger)
        autoencoder.build_model()
        
        # 生成训练数据（正常数据）
        np.random.seed(42)
        training_data = np.random.normal(0, 1, (100, self.config['input_features']))
        
        # 训练模型
        training_result = autoencoder.train(training_data, epochs=5)
        self.assertIsInstance(training_result, dict)
        self.assertIn('final_loss', training_result)
        
        # 测试预测
        test_data = np.random.normal(0, 1, (1, self.config['input_features']))
        prediction_result = autoencoder.predict(test_data)
        
        self.assertIsInstance(prediction_result, dict)
        self.assertIn('is_anomaly', prediction_result)
        self.assertIn('reconstruction_error', prediction_result)


class TestErrorClassifier(unittest.TestCase):
    """测试错误分类器"""
    
    def setUp(self):
        self.logger = SystemLogger({
            'level': 'INFO',
            'file_path': '/tmp/test_classifier.log',
            'console_output': False
        })
        
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model_path': os.path.join(self.temp_dir, 'test_classifier.pkl'),
            'classes': ['signal_interference', 'bandwidth_congestion', 'authentication_failure', 
                       'packet_corruption', 'dns_resolution_error', 'gateway_unreachable',
                       'memory_leak', 'cpu_overload'],
            'confidence_threshold': 0.7
        }
    
    def test_classifier_building(self):
        """测试分类器构建"""
        classifier = ErrorClassifier(self.config, self.logger)
        model = classifier.build_classifier()
        
        self.assertIsNotNone(model)
        self.assertEqual(len(classifier.classes), 8)
    
    def test_training_and_classification(self):
        """测试分类器训练和分类"""
        classifier = ErrorClassifier(self.config, self.logger)
        classifier.build_classifier()
        
        # 生成训练数据
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        training_data = np.random.normal(0, 1, (n_samples, n_features))
        training_labels = np.random.choice(classifier.classes, n_samples)
        
        # 训练分类器
        training_result = classifier.train(training_data, training_labels.tolist())
        self.assertIsInstance(training_result, dict)
        self.assertIn('accuracy', training_result)
        
        # 测试分类
        test_data = np.random.normal(0, 1, (1, n_features))
        classification_result = classifier.classify_error(test_data)
        
        self.assertIsInstance(classification_result, dict)
        self.assertIn('predicted_class', classification_result)
        self.assertIn('confidence', classification_result)


class TestAnomalyDetectionEngine(unittest.TestCase):
    """测试异常检测引擎"""
    
    def setUp(self):
        self.logger = SystemLogger({
            'level': 'INFO',
            'file_path': '/tmp/test_engine.log',
            'console_output': False
        })
        
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟的组件
        autoencoder_config = {
            'input_features': 10,
            'encoding_dim': 5,
            'threshold': 0.02,
            'model_path': os.path.join(self.temp_dir, 'autoencoder.h5')
        }
        
        classifier_config = {
            'model_path': os.path.join(self.temp_dir, 'classifier.pkl'),
            'classes': ['signal_interference', 'bandwidth_congestion', 'authentication_failure', 
                       'packet_corruption', 'dns_resolution_error', 'gateway_unreachable'],
            'confidence_threshold': 0.7
        }
        
        buffer_config = {
            'buffer_size': 100,
            'anomaly_data_path': self.temp_dir
        }
        
        engine_config = {
            'detection_window': 5,
            'severity_levels': {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        }
        
        self.autoencoder = AutoencoderModel(autoencoder_config, self.logger)
        self.classifier = ErrorClassifier(classifier_config, self.logger)
        self.buffer = CircularBuffer(buffer_config, self.logger)
        
        self.engine = AnomalyDetectionEngine(
            engine_config, self.autoencoder, self.classifier, self.buffer, self.logger
        )
    
    def test_anomaly_detection_workflow(self):
        """测试完整的异常检测工作流"""
        # 准备测试数据
        test_data = []
        for i in range(5):
            data = {
                'wlan0_signal_strength': -50.0 + i,
                'packet_loss_rate': 0.01 * i,
                'cpu_percent': 20.0 + i,
                'memory_percent': 50.0 + i,
                'timestamp': f'2023-01-01T00:0{i}:00'
            }
            test_data.append(data)
        
        # 执行异常检测
        result = self.engine.detect_anomaly(test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_anomaly', result)
        self.assertIn('timestamp', result)
    
    def test_detection_statistics(self):
        """测试检测统计功能"""
        # 执行几次检测
        for i in range(3):
            test_data = [{'value': i, 'metric': i * 10}]
            self.engine.detect_anomaly(test_data)
        
        stats = self.engine.get_detection_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_detections', stats)
        self.assertEqual(stats['total_detections'], 3)


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建临时配置文件
        self.config = {
            "system": {
                "name": "测试系统",
                "version": "1.0.0"
            },
            "data_collection": {
                "interfaces": ["lo"],
                "collection_interval": 1,
                "metrics": ["throughput", "connection_count"]
            },
            "buffer_manager": {
                "buffer_size": 50,
                "anomaly_data_path": self.temp_dir
            },
            "ai_models": {
                "autoencoder": {
                    "input_features": 10,
                    "encoding_dim": 5,
                    "threshold": 0.02,
                    "model_path": os.path.join(self.temp_dir, "autoencoder.h5")
                },
                "classifier": {
                    "model_path": os.path.join(self.temp_dir, "classifier.pkl"),
                    "classes": ["signal_interference", "bandwidth_congestion", "authentication_failure", 
                               "packet_corruption", "dns_resolution_error", "gateway_unreachable"],
                    "confidence_threshold": 0.7
                }
            },
            "anomaly_detection": {
                "detection_window": 5,
                "severity_levels": {"low": 0.3, "medium": 0.6, "high": 0.9}
            },
            "logging": {
                "level": "INFO",
                "file_path": os.path.join(self.temp_dir, "test.log"),
                "console_output": False
            }
        }
        
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def test_full_system_initialization(self):
        """测试完整系统初始化"""
        # 导入主应用
        from src.main import NetworkAnomalyDetectorApp
        
        try:
            app = NetworkAnomalyDetectorApp(self.config_file)
            self.assertIsNotNone(app)
            
            # 检查各组件是否正确初始化
            self.assertIsNotNone(app.data_collector)
            self.assertIsNotNone(app.feature_extractor)
            self.assertIsNotNone(app.buffer_manager)
            self.assertIsNotNone(app.autoencoder)
            self.assertIsNotNone(app.error_classifier)
            self.assertIsNotNone(app.anomaly_engine)
            
            # 获取系统状态
            status = app.get_system_status()
            self.assertIsInstance(status, dict)
            self.assertIn('running', status)
            
        except Exception as e:
            self.fail(f"系统初始化失败: {e}")


def run_performance_tests():
    """运行性能测试"""
    print("\n=== 性能测试 ===")
    
    logger = SystemLogger({
        'level': 'ERROR',
        'file_path': '/tmp/perf_test.log',
        'console_output': False
    })
    
    # 测试特征提取性能
    print("测试特征提取性能...")
    extractor = FeatureExtractor(['signal_strength', 'packet_loss_rate'], logger)
    
    import time
    start_time = time.time()
    
    for i in range(1000):
        raw_data = {
            'wlan0_signal_strength': -50.0 + np.random.normal(0, 5),
            'packet_loss_rate': max(0, np.random.normal(0.01, 0.005)),
            'cpu_percent': 20 + np.random.normal(0, 10)
        }
        features = extractor.extract_features(raw_data)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 1000
    print(f"特征提取平均耗时: {avg_time*1000:.2f} ms")
    
    # 测试缓冲区性能
    print("测试缓冲区性能...")
    buffer_config = {
        'buffer_size': 1000,
        'anomaly_data_path': '/tmp'
    }
    buffer = CircularBuffer(buffer_config, logger)
    
    start_time = time.time()
    
    for i in range(10000):
        data = {'value': i, 'timestamp': time.time()}
        buffer.add_data(data)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10000
    print(f"缓冲区操作平均耗时: {avg_time*1000:.4f} ms")


def main():
    """主测试函数"""
    print("AI异常检测系统测试套件")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestSystemLogger,
        TestNetworkDataCollector,
        TestFeatureExtractor,
        TestCircularBuffer,
        TestAutoencoder,
        TestErrorClassifier,
        TestAnomalyDetectionEngine,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 运行性能测试
    run_performance_tests()
    
    # 输出测试结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 