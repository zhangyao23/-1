#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
错误分类器模块

当检测到网络异常后，使用机器学习分类器确定具体的异常类型。
支持beacon_stuck、disconnection、fail_to_connect、data_stall等异常分类。
"""

import numpy as np
import joblib
import os
from typing import Dict, List, Any
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class ErrorClassifier:
    """
    网络错误分类器
    
    基于随机森林的多分类器，用于识别具体的网络异常类型。
    """
    
    def __init__(self, config: Dict, logger):
        """
        初始化错误分类器
        
        Args:
            config (Dict): 分类器配置参数
            logger: 系统日志记录器
        """
        self.config = config
        self.logger = logger
        
        # 配置参数
        self.model_path = config.get('model_path', 'models/error_classifier.pkl')
        self.classes = config.get('classes', [
            'signal_interference', 'bandwidth_congestion', 'authentication_failure', 
            'packet_corruption', 'dns_resolution_error', 'gateway_unreachable',
            'memory_leak', 'cpu_overload'
        ])
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # 新增：从配置中读取可调超参数
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 10)
        self.min_samples_split = config.get('min_samples_split', 5)
        
        # 模型组件
        self.classifier = None
        self.label_encoder = LabelEncoder()
        
        # 统计信息
        self.classification_count = 0
        self.class_predictions = {cls: 0 for cls in self.classes}
        
        # 加载预训练模型
        self._load_model()
        
        self.logger.info(f"错误分类器初始化完成，支持类别: {self.classes}")
    
    def build_classifier(self) -> RandomForestClassifier:
        """构建随机森林分类器"""
        try:
            self.classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42,
                class_weight='balanced'
            )
            
            self.label_encoder.fit(self.classes)
            self.logger.info("随机森林分类器构建完成")
            return self.classifier
            
        except Exception as e:
            self.logger.error(f"构建分类器失败: {e}")
            raise
    
    def train(self, training_data: np.ndarray, training_labels: List[str]) -> Dict[str, Any]:
        """
        训练错误分类器
        
        Args:
            training_data: 训练特征数据
            training_labels: 训练标签
            
        Returns:
            训练结果
        """
        try:
            self.logger.info(f"开始训练错误分类器，数据形状: {training_data.shape}")
            
            # 新增: 清洗标签，去除首尾可能存在的空格
            cleaned_labels = [label.strip() for label in training_labels]
            
            if self.classifier is None:
                self.build_classifier()
            
            # 编码标签并训练
            encoded_labels = self.label_encoder.transform(cleaned_labels)
            self.classifier.fit(training_data, encoded_labels)
            
            # 计算准确率
            accuracy = self.classifier.score(training_data, encoded_labels)
            
            # 保存模型
            self._save_model()
            
            result = {
                'accuracy': accuracy,
                'training_samples': len(training_data),
                'classes': self.classes,
                'model_saved': True
            }
            
            self.logger.info(f"分类器训练完成，准确率: {accuracy:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"分类器训练失败: {e}")
            raise
    
    def classify_error(self, features: np.ndarray) -> Dict[str, Any]:
        """
        对异常特征进行分类
        
        Args:
            features: 特征向量
            
        Returns:
            分类结果
        """
        try:
            if self.classifier is None:
                return self._get_default_classification()
            
            self.classification_count += 1
            
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # 预测类别和概率
            predicted_class_idx = self.classifier.predict(features)[0]
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            class_probabilities = self.classifier.predict_proba(features)[0]
            confidence = np.max(class_probabilities)
            
            # 更新统计
            self.class_predictions[predicted_class] += 1
            
            result = {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'is_confident': confidence >= self.confidence_threshold,
                'class_probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.classes, class_probabilities)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.debug(f"错误分类: {predicted_class} (置信度: {confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"错误分类失败: {e}")
            return self._get_default_classification()
    
    def _get_default_classification(self) -> Dict[str, Any]:
        """获取默认分类结果"""
        return {
            'predicted_class': 'unknown',
            'confidence': 0.0,
            'is_confident': False,
            'class_probabilities': {cls: 0.0 for cls in self.classes},
            'timestamp': datetime.now().isoformat()
        }
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """获取分类统计信息"""
        stats = {
            'total_classifications': self.classification_count,
            'classes': self.classes,
            'class_predictions': self.class_predictions.copy(),
            'confidence_threshold': self.confidence_threshold,
            'model_loaded': self.classifier is not None
        }
        
        if self.classification_count > 0:
            stats['class_distribution'] = {
                cls: count / self.classification_count * 100 
                for cls, count in self.class_predictions.items()
            }
        
        return stats
    
    def _save_model(self):
        """保存分类器模型"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'classes': self.classes,
                'confidence_threshold': self.confidence_threshold,
                'class_predictions': self.class_predictions,
                'classification_count': self.classification_count,
                'saved_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, self.model_path)
            self.logger.info(f"分类器模型已保存到: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"保存分类器模型失败: {e}")
    
    def _load_model(self):
        """加载分类器模型"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.info("未找到预训练分类器模型")
                return
            
            model_data = joblib.load(self.model_path)
            
            self.classifier = model_data.get('classifier')
            self.label_encoder = model_data.get('label_encoder', LabelEncoder())
            self.classes = model_data.get('classes', self.classes)
            self.confidence_threshold = model_data.get('confidence_threshold', self.confidence_threshold)
            self.class_predictions = model_data.get('class_predictions', {cls: 0 for cls in self.classes})
            self.classification_count = model_data.get('classification_count', 0)
            
            self.logger.info(f"分类器模型已从 {self.model_path} 加载")
            
        except Exception as e:
            self.logger.warning(f"加载分类器模型失败: {e}")
            self.classifier = None


def create_error_classifier(config: Dict, logger) -> ErrorClassifier:
    """创建错误分类器实例"""
    return ErrorClassifier(config, logger) 