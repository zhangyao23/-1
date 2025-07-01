#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
异常检测引擎

整合自编码器异常检测和错误分类器，实现完整的异常检测工作流。
协调各个组件，提供统一的异常检测接口。

工作流程：
1. 接收特征数据
2. 使用自编码器检测是否存在异常
3. 如果检测到异常，使用分类器确定异常类型
4. 返回完整的检测结果
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


class AnomalyDetectionEngine:
    """
    异常检测引擎
    
    协调自编码器和错误分类器，提供完整的异常检测功能。
    包含异常检测、类型分类、结果评估等核心功能。
    """
    
    def __init__(self, config: Dict, autoencoder, error_classifier, buffer_manager, logger):
        """
        初始化异常检测引擎
        
        Args:
            config: 检测引擎配置
            autoencoder: 自编码器模型
            error_classifier: 错误分类器
            buffer_manager: 缓冲区管理器
            logger: 日志记录器
        """
        self.config = config
        self.autoencoder = autoencoder
        self.error_classifier = error_classifier
        self.buffer_manager = buffer_manager
        self.logger = logger
        
        # 配置参数
        self.detection_window = config.get('detection_window', 10)
        self.severity_levels = config.get('severity_levels', {
            'low': 0.3, 'medium': 0.6, 'high': 0.9
        })
        
        # 统计信息
        self.total_detections = 0
        self.anomaly_detections = 0
        self.detection_history = []
        
        # 性能指标
        self.avg_detection_time = 0.0
        self.detection_times = []
        
        self.logger.info("异常检测引擎初始化完成")
    
    def detect_anomaly_from_vector(self, feature_vector: np.ndarray, feature_names: List[str]) -> (bool, Dict):
        """
        [模拟器专用] 直接从特征向量进行异常检测。
        如果注入了真实的自编码器和分类器模型，则使用它们；
        否则，回退到简化的模拟判断逻辑。
        """
        # --- 优先使用真实模型逻辑 ---
        if self.autoencoder and hasattr(self.autoencoder, 'predict'):
            self.logger.info("检测到真实模型，正在使用自编码器进行预测...")
            
            # 确保输入是2D的
            if feature_vector.ndim == 1:
                feature_vector_2d = feature_vector.reshape(1, -1)
            else:
                feature_vector_2d = feature_vector
                
            autoencoder_result = self.autoencoder.predict(feature_vector_2d)
            is_anomaly = autoencoder_result.get('is_anomaly', False) # 直接获取布尔值
            details = autoencoder_result
            
            # 安全地提取重构误差，无论其返回的是单个浮点数还是数组
            recon_error_val = details.get('reconstruction_error', 0.0)
            if isinstance(recon_error_val, (list, np.ndarray)) and len(recon_error_val) > 0:
                details['reconstruction_error'] = float(recon_error_val[0])
            else:
                details['reconstruction_error'] = float(recon_error_val)

            details['is_anomaly'] = bool(is_anomaly)

            if is_anomaly and self.error_classifier and hasattr(self.error_classifier, 'classify_error'):
                self.logger.info("自编码器发现异常，正在使用错误分类器进行分类...")
                classification_result = self.error_classifier.classify_error(feature_vector_2d)
                details.update(classification_result)
            
            return is_anomaly, details

        # --- Fallback to Simulation Logic ---
        self.logger.warning("未检测到真实模型，回退到简化的模拟判断逻辑。")
        # 对于一个已经标准化的向量（正常样本被转换为全零向量），
        # 其L2范数代表了它与"正常"中心的距离。
        anomaly_score = np.linalg.norm(feature_vector)
        
        # 这是一个经验阈值。在标准化空间中，距离大于某个值可被认为是异常。
        # 我们基于特征向量的维度动态计算该阈值，使其更具鲁棒性。
        # 核心思想是，如果每个特征都偏离一个标准差，总距离就是sqrt(n)。
        threshold = np.sqrt(len(feature_vector)) * 1.5  # 设定为1.5个标准差的距离

        is_anomaly = anomaly_score > threshold

        details = {
            "reason": "Detected via simplified vector norm check (simulation only).",
            "anomaly_score": anomaly_score,
            "threshold": threshold
        }

        if is_anomaly:
            # 模拟一个简单的分类
            try:
                cpu_index = feature_names.index('cpu_usage')
                latency_index = feature_names.index('avg_latency')

                if feature_vector[cpu_index] > 1.0:
                     details['predicted_class'] = 'cpu_overload'
                elif feature_vector[latency_index] > 1.0:
                     details['predicted_class'] = 'high_latency'
                else:
                     details['predicted_class'] = 'unknown_anomaly'
            except (ValueError, IndexError):
                details['predicted_class'] = 'unknown_anomaly (feature name not found)'

        return is_anomaly, details
    
    def detect_anomaly(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行异常检测
        
        Args:
            data: 要检测的数据列表
            
        Returns:
            异常检测结果
        """
        start_time = time.time()
        
        try:
            self.total_detections += 1
            
            # 准备检测数据
            if not data:
                return self._create_no_anomaly_result("数据为空")
            
            # 提取最新的特征数据
            latest_features = self._extract_features_from_data(data)
            if latest_features is None or len(latest_features) == 0:
                return self._create_no_anomaly_result("特征提取失败")
            
            # 第一步：使用自编码器检测异常
            autoencoder_result = self.autoencoder.predict(latest_features)
            
            # 检查是否检测到异常
            if not autoencoder_result.get('is_anomaly', False):
                return self._create_no_anomaly_result("未检测到异常", autoencoder_result)
            
            # 第二步：如果检测到异常，进行分类
            classification_result = self.error_classifier.classify_error(latest_features)
            
            # 第三步：评估异常严重程度
            severity = self._evaluate_severity(autoencoder_result, classification_result)
            
            # 构建完整的异常检测结果
            anomaly_result = self._create_anomaly_result(
                autoencoder_result, 
                classification_result, 
                severity,
                data
            )
            
            # 更新统计信息
            self.anomaly_detections += 1
            self._update_detection_history(anomaly_result)
            
            # 记录检测时间
            detection_time = time.time() - start_time
            self._update_performance_metrics(detection_time)
            
            self.logger.warning(f"检测到异常: {anomaly_result['anomaly_type']}")
            return anomaly_result
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            return self._create_error_result(str(e))
    
    def _extract_features_from_data(self, data: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        从数据中提取特征
        
        Args:
            data: 原始数据列表
            
        Returns:
            提取的特征向量
        """
        try:
            if not data:
                return None
            
            # 使用最新的数据点
            latest_data = data[-1]
            
            # 提取数值特征
            features = []
            for key, value in latest_data.items():
                if key in ['timestamp', 'buffer_timestamp', 'buffer_datetime']:
                    continue
                
                try:
                    if isinstance(value, (int, float)):
                        features.append(float(value))
                    elif isinstance(value, bool):
                        features.append(float(value))
                    else:
                        features.append(0.0)
                except:
                    features.append(0.0)
            
            if not features:
                return None
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            return None
    
    def _evaluate_severity(self, autoencoder_result: Dict, classification_result: Dict) -> str:
        """
        评估异常严重程度
        
        Args:
            autoencoder_result: 自编码器检测结果
            classification_result: 分类结果
            
        Returns:
            严重程度级别
        """
        try:
            # 基于重构误差和分类置信度评估严重程度
            reconstruction_error = autoencoder_result.get('reconstruction_error', 0.0)
            classification_confidence = classification_result.get('confidence', 0.0)
            
            # 计算综合严重程度分数
            severity_score = (reconstruction_error + classification_confidence) / 2.0
            
            # 根据阈值判断严重程度
            if severity_score >= self.severity_levels['high']:
                return 'high'
            elif severity_score >= self.severity_levels['medium']:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"严重程度评估失败: {e}")
            return 'low'
    
    def _create_anomaly_result(self, autoencoder_result: Dict, classification_result: Dict, 
                              severity: str, original_data: List[Dict]) -> Dict[str, Any]:
        """
        创建异常检测结果
        
        Args:
            autoencoder_result: 自编码器结果
            classification_result: 分类结果  
            severity: 严重程度
            original_data: 原始数据
            
        Returns:
            完整的异常检测结果
        """
        return {
            'is_anomaly': True,
            'anomaly_type': classification_result.get('predicted_class', 'unknown'),
            'confidence': classification_result.get('confidence', 0.0),
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'detection_details': {
                'reconstruction_error': autoencoder_result.get('reconstruction_error', 0.0),
                'anomaly_threshold': autoencoder_result.get('threshold', 0.0),
                'anomaly_score': autoencoder_result.get('anomaly_score', 0.0),
                'classification_probabilities': classification_result.get('class_probabilities', {}),
                'is_classification_confident': classification_result.get('is_confident', False)
            },
            'data_context': {
                'detection_window_size': len(original_data),
                'latest_data_timestamp': original_data[-1].get('buffer_datetime', 'Unknown') if original_data else 'Unknown'
            },
            'engine_stats': {
                'total_detections': self.total_detections,
                'anomaly_rate': self.anomaly_detections / max(self.total_detections, 1) * 100
            }
        }
    
    def _create_no_anomaly_result(self, reason: str = "", autoencoder_result: Dict = None) -> Dict[str, Any]:
        """
        创建无异常检测结果
        
        Args:
            reason: 无异常的原因
            autoencoder_result: 自编码器结果（可选）
            
        Returns:
            无异常检测结果
        """
        result = {
            'is_anomaly': False,
            'anomaly_type': 'none',
            'confidence': 0.0,
            'severity': 'none',
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        }
        
        if autoencoder_result:
            result['detection_details'] = {
                'reconstruction_error': autoencoder_result.get('reconstruction_error', 0.0),
                'anomaly_threshold': autoencoder_result.get('threshold', 0.0),
                'anomaly_score': autoencoder_result.get('anomaly_score', 0.0)
            }
        
        return result
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        创建错误检测结果
        
        Args:
            error_message: 错误信息
            
        Returns:
            错误检测结果
        """
        return {
            'is_anomaly': False,
            'anomaly_type': 'detection_error',
            'confidence': 0.0,
            'severity': 'none',
            'timestamp': datetime.now().isoformat(),
            'error': error_message
        }
    
    def _update_detection_history(self, result: Dict[str, Any]):
        """
        更新检测历史记录
        
        Args:
            result: 检测结果
        """
        try:
            # 保持历史记录大小
            max_history = 100
            
            history_entry = {
                'timestamp': result['timestamp'],
                'is_anomaly': result['is_anomaly'],
                'anomaly_type': result.get('anomaly_type', 'none'),
                'confidence': result.get('confidence', 0.0),
                'severity': result.get('severity', 'none')
            }
            
            self.detection_history.append(history_entry)
            
            # 保持历史记录在限制范围内
            if len(self.detection_history) > max_history:
                self.detection_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"更新检测历史失败: {e}")
    
    def _update_performance_metrics(self, detection_time: float):
        """
        更新性能指标
        
        Args:
            detection_time: 检测耗时
        """
        try:
            self.detection_times.append(detection_time)
            
            # 保持最近的检测时间记录
            if len(self.detection_times) > 100:
                self.detection_times.pop(0)
            
            # 计算平均检测时间
            self.avg_detection_time = np.mean(self.detection_times)
            
        except Exception as e:
            self.logger.error(f"更新性能指标失败: {e}")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Returns:
            检测统计数据
        """
        stats = {
            'total_detections': self.total_detections,
            'anomaly_detections': self.anomaly_detections,
            'anomaly_rate': self.anomaly_detections / max(self.total_detections, 1) * 100,
            'avg_detection_time_ms': self.avg_detection_time * 1000,
            'detection_window': self.detection_window,
            'severity_levels': self.severity_levels
        }
        
        # 最近检测历史统计
        if self.detection_history:
            recent_anomalies = sum(1 for h in self.detection_history[-10:] if h['is_anomaly'])
            stats['recent_anomaly_rate'] = recent_anomalies / min(10, len(self.detection_history)) * 100
            
            # 按类型统计
            anomaly_types = {}
            for history in self.detection_history:
                if history['is_anomaly']:
                    anomaly_type = history['anomaly_type']
                    anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            stats['anomaly_types_distribution'] = anomaly_types
        
        return stats
    
    def batch_detect(self, data_batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        批量异常检测
        
        Args:
            data_batches: 数据批次列表
            
        Returns:
            检测结果列表
        """
        results = []
        
        for batch in data_batches:
            try:
                result = self.detect_anomaly(batch)
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量检测失败: {e}")
                results.append(self._create_error_result(str(e)))
        
        self.logger.info(f"批量检测完成，处理了 {len(data_batches)} 个批次")
        return results
    
    def set_detection_sensitivity(self, sensitivity: str):
        """
        调整检测敏感度
        
        Args:
            sensitivity: 敏感度级别 ('low', 'medium', 'high')
        """
        try:
            sensitivity_configs = {
                'low': {'threshold_multiplier': 1.5, 'confidence_threshold': 0.8},
                'medium': {'threshold_multiplier': 1.0, 'confidence_threshold': 0.7},
                'high': {'threshold_multiplier': 0.7, 'confidence_threshold': 0.6}
            }
            
            if sensitivity in sensitivity_configs:
                config = sensitivity_configs[sensitivity]
                
                # 调整自编码器阈值
                current_threshold = self.autoencoder.threshold
                new_threshold = current_threshold * config['threshold_multiplier']
                self.autoencoder.update_threshold(new_threshold)
                
                # 调整分类器置信度阈值
                self.error_classifier.update_confidence_threshold(config['confidence_threshold'])
                
                self.logger.info(f"检测敏感度已调整为: {sensitivity}")
            else:
                self.logger.warning(f"未知的敏感度级别: {sensitivity}")
                
        except Exception as e:
            self.logger.error(f"调整检测敏感度失败: {e}")
    
    def export_detection_report(self, filepath: str) -> bool:
        """
        导出检测报告
        
        Args:
            filepath: 报告文件路径
            
        Returns:
            导出是否成功
        """
        try:
            import json
            
            report = {
                'report_generated_at': datetime.now().isoformat(),
                'detection_statistics': self.get_detection_stats(),
                'autoencoder_summary': self.autoencoder.get_model_summary(),
                'classifier_summary': self.error_classifier.get_classification_stats(),
                'detection_history': self.detection_history[-50:],  # 最近50条记录
                'configuration': self.config
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"检测报告已导出到: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出检测报告失败: {e}")
            return False


def create_anomaly_detection_engine(config: Dict, autoencoder, error_classifier, 
                                   buffer_manager, logger) -> AnomalyDetectionEngine:
    """
    创建异常检测引擎实例
    
    Args:
        config: 配置参数
        autoencoder: 自编码器
        error_classifier: 错误分类器
        buffer_manager: 缓冲区管理器
        logger: 日志记录器
        
    Returns:
        异常检测引擎实例
    """
    return AnomalyDetectionEngine(config, autoencoder, error_classifier, buffer_manager, logger) 