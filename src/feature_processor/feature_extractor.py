#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特征提取和处理模块

将从网络接口收集的原始数据转换为适合AI模型处理的特征向量。
包含数据清洗、归一化、特征工程和维度处理等功能。

主要功能：
1. 原始数据清洗和验证
2. 特征提取和计算
3. 数据归一化和标准化  
4. 时间序列特征生成
5. 异常值检测和处理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import json
import joblib
import os


class FeatureExtractor:
    """
    网络数据特征提取器
    
    负责将原始网络监控数据转换为标准化的特征向量，
    为后续的异常检测模型提供高质量的输入数据。
    """
    
    def __init__(self, metrics_config: List[str], logger, scaler_path: Optional[str] = None):
        """
        初始化特征提取器
        
        Args:
            metrics_config (List[str]): 需要处理的指标列表
            logger: 系统日志记录器
            scaler_path: 预训练scaler的路径，如果为None则创建新的scaler
        """
        self.metrics_config = metrics_config
        self.logger = logger
        self.scaler_path = scaler_path or "models/autoencoder_model/autoencoder_scaler.pkl"
        
        # 初始化数据预处理器
        self._initialize_scaler()
        
        # 特征名称映射
        self.feature_names = []
        
        # 历史数据窗口，用于计算时间序列特征
        self.history_window = 10
        self.historical_data = []
        
        # 特征统计信息
        self.feature_stats = {}
        
        self.logger.info("特征提取器初始化完成")
    
    def _initialize_scaler(self):
        """初始化或加载标准化器"""
        try:
            if os.path.exists(self.scaler_path):
                # 加载预训练的scaler
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info(f"已加载预训练的标准化器: {self.scaler_path}")
                self._use_pretrained_scaler = True
            else:
                # 创建新的scaler（用于训练模式）
                self.scaler = StandardScaler()
                self.logger.info("创建新的标准化器")
                self._use_pretrained_scaler = False
        except Exception as e:
            self.logger.warning(f"加载预训练scaler失败: {e}, 创建新的scaler")
            self.scaler = StandardScaler()
            self._use_pretrained_scaler = False
    
    def extract_features(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        从原始数据中提取特征向量
        
        Args:
            raw_data (Dict[str, Any]): 原始网络监控数据
            
        Returns:
            np.ndarray: 标准化的特征向量
        """
        try:
            # 数据清洗和验证
            cleaned_data = self._clean_raw_data(raw_data)
            
            # 基础特征提取
            basic_features = self._extract_basic_features(cleaned_data)
            
            # 统计特征计算
            statistical_features = self._calculate_statistical_features(cleaned_data)
            
            # 时间序列特征
            temporal_features = self._extract_temporal_features(cleaned_data)
            
            # 合并所有特征
            all_features = {**basic_features, **statistical_features, **temporal_features}
            
            # 转换为特征向量
            feature_vector = self._convert_to_vector(all_features)
            
            # 数据归一化
            normalized_features = self._normalize_features(feature_vector)
            
            # 更新历史数据
            self._update_history(cleaned_data)
            
            self.logger.debug(f"特征提取完成，维度: {len(normalized_features)}")
            return normalized_features
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            return np.array([])
    
    def _clean_raw_data(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """
        清洗和验证原始数据
        
        Args:
            raw_data (Dict[str, Any]): 原始数据
            
        Returns:
            Dict[str, float]: 清洗后的数据
        """
        cleaned_data = {}
        
        for key, value in raw_data.items():
            try:
                # 跳过非数值字段
                if key in ['timestamp', 'collection_time']:
                    continue
                
                # 转换为数值类型
                if isinstance(value, (int, float)):
                    # 处理异常值
                    if np.isnan(value) or np.isinf(value):
                        cleaned_data[key] = 0.0
                    else:
                        cleaned_data[key] = float(value)
                elif isinstance(value, bool):
                    cleaned_data[key] = float(value)
                elif isinstance(value, str):
                    # 尝试转换字符串为数值
                    try:
                        cleaned_data[key] = float(value)
                    except ValueError:
                        cleaned_data[key] = 0.0
                else:
                    cleaned_data[key] = 0.0
                    
            except Exception as e:
                self.logger.debug(f"数据清洗失败 {key}: {e}")
                cleaned_data[key] = 0.0
        
        return cleaned_data
    
    def _extract_basic_features(self, data: Dict[str, float]) -> Dict[str, float]:
        """
        提取基础网络特征
        
        Args:
            data (Dict[str, float]): 清洗后的数据
            
        Returns:
            Dict[str, float]: 基础特征
        """
        basic_features = {}
        
        # 信号强度相关特征 (只使用quality，不包含level)
        quality_keys = [k for k in data.keys() if 'quality' in k.lower()]
        if quality_keys:
            quality_values = [data[k] for k in quality_keys]
            basic_features['avg_signal_strength'] = np.mean(quality_values)
        else:
            # 如果没有质量指标，使用所有信号相关指标（向后兼容）
            signal_keys = [k for k in data.keys() if 'signal' in k.lower()]
        if signal_keys:
            signal_values = [data[k] for k in signal_keys]
            basic_features['avg_signal_strength'] = np.mean(signal_values)
        
        # 数据传输速率特征 (转换为Mbps)
        rate_keys = [k for k in data.keys() if 'rate' in k.lower() and 'bps' in k.lower()]
        if rate_keys:
            rate_values = [data[k] / 1000000.0 for k in rate_keys]  # 转换为Mbps
            basic_features['total_data_rate'] = np.sum(rate_values)
            basic_features['avg_data_rate'] = np.mean(rate_values)
            basic_features['max_data_rate'] = np.max(rate_values)
        
        # 丢包率特征
        loss_keys = [k for k in data.keys() if 'loss' in k.lower() or 'drop' in k.lower()]
        if loss_keys:
            loss_values = [data[k] for k in loss_keys]
            basic_features['total_packet_loss'] = np.sum(loss_values)
            basic_features['max_packet_loss'] = np.max(loss_values)
        
        # 延迟特征 (包含所有延迟相关指标)
        latency_keys = [k for k in data.keys() if 'latency' in k.lower() or 'ping' in k.lower() or 'response_time' in k.lower()]
        if latency_keys:
            latency_values = [data[k] for k in latency_keys]
            basic_features['avg_latency'] = np.mean(latency_values)
            basic_features['max_latency'] = np.max(latency_values)
        
        # 连接数特征
        connection_keys = [k for k in data.keys() if 'connection' in k.lower()]
        if connection_keys:
            connection_values = [data[k] for k in connection_keys]
            basic_features['total_connections'] = np.sum(connection_values)
        
        # 系统资源特征
        if 'cpu_percent' in data:
            basic_features['cpu_usage'] = data['cpu_percent']
        if 'memory_percent' in data:
            basic_features['memory_usage'] = data['memory_percent']
        
        return basic_features
    
    def _calculate_statistical_features(self, data: Dict[str, float]) -> Dict[str, float]:
        """
        计算统计特征
        
        Args:
            data (Dict[str, float]): 输入数据
            
        Returns:
            Dict[str, float]: 统计特征
        """
        statistical_features = {}
        
        # 按接口分组计算统计特征
        interfaces = ['wlan0', 'eth0']
        
        for interface in interfaces:
            interface_data = [v for k, v in data.items() if interface in k]
            
            if interface_data:
                statistical_features[f'{interface}_mean'] = np.mean(interface_data)
                statistical_features[f'{interface}_std'] = np.std(interface_data)
                statistical_features[f'{interface}_median'] = np.median(interface_data)
                statistical_features[f'{interface}_range'] = np.max(interface_data) - np.min(interface_data)
        
        # 全局统计特征
        all_values = list(data.values())
        if all_values:
            statistical_features['global_mean'] = np.mean(all_values)
            statistical_features['global_std'] = np.std(all_values)
            statistical_features['global_skewness'] = self._calculate_skewness(all_values)
            statistical_features['global_kurtosis'] = self._calculate_kurtosis(all_values)
        
        return statistical_features
    
    def _extract_temporal_features(self, current_data: Dict[str, float]) -> Dict[str, float]:
        """
        提取时间序列特征
        
        Args:
            current_data (Dict[str, float]): 当前数据
            
        Returns:
            Dict[str, float]: 时间序列特征
        """
        temporal_features = {}
        
        if len(self.historical_data) < 2:
            # 历史数据不足，返回默认值
            temporal_features['trend'] = 0.0
            temporal_features['volatility'] = 0.0
            temporal_features['momentum'] = 0.0
            return temporal_features
        
        try:
            # 计算关键指标的趋势
            key_metrics = ['avg_signal_strength', 'total_data_rate', 'total_packet_loss', 'avg_latency']
            
            for metric in key_metrics:
                if metric in current_data:
                    # 获取历史值
                    historical_values = []
                    for hist_data in self.historical_data[-5:]:  # 最近5个数据点
                        if metric in hist_data:
                            historical_values.append(hist_data[metric])
                    
                    if len(historical_values) >= 2:
                        # 计算趋势（线性回归斜率）
                        trend = self._calculate_trend(historical_values)
                        temporal_features[f'{metric}_trend'] = trend
                        
                        # 计算波动性
                        volatility = np.std(historical_values)
                        temporal_features[f'{metric}_volatility'] = volatility
                        
                        # 计算动量（最近值与平均值的比值）
                        if np.mean(historical_values) != 0:
                            momentum = current_data[metric] / np.mean(historical_values)
                            temporal_features[f'{metric}_momentum'] = momentum
            
            # 变化率特征
            if len(self.historical_data) >= 1:
                last_data = self.historical_data[-1]
                for key in current_data:
                    if key in last_data and last_data[key] != 0:
                        change_rate = (current_data[key] - last_data[key]) / abs(last_data[key])
                        temporal_features[f'{key}_change_rate'] = change_rate
            
        except Exception as e:
            self.logger.debug(f"时间序列特征计算失败: {e}")
        
        return temporal_features
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        计算数据趋势（简单线性回归斜率）
        
        Args:
            values (List[float]): 数值序列
            
        Returns:
            float: 趋势斜率
        """
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.arange(len(values))
            y = np.array(values)
            
            # 计算线性回归斜率
            n = len(values)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            
            return slope
        except Exception:
            return 0.0
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """计算偏度"""
        try:
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                return 0.0
            
            skewness = np.mean(((values - mean) / std) ** 3)
            return skewness
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """计算峰度"""
        try:
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                return 0.0
            
            kurtosis = np.mean(((values - mean) / std) ** 4) - 3
            return kurtosis
        except Exception:
            return 0.0
    
    def _convert_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        将特征字典转换为6维向量（降维处理）
        
        Args:
            features (Dict[str, float]): 特征字典
            
        Returns:
            np.ndarray: 6维特征向量
        """
        # 定义固定的6个最重要特征名称，确保维度一致性
        if not self.feature_names:
            self.feature_names = [
                'avg_signal_strength',  # 信号强度
                'avg_data_rate',        # 数据传输速率
                'avg_latency',          # 网络延迟
                'total_packet_loss',    # 丢包率
                'cpu_usage',            # CPU使用率
                'memory_usage'          # 内存使用率
            ]
        
        # 按固定顺序提取特征值，始终保持6维
        feature_vector = []
        for name in self.feature_names:
            value = features.get(name, 0.0)
            feature_vector.append(value)
        
        # 确保正好是6维
        while len(feature_vector) < 6:
            feature_vector.append(0.0)
        feature_vector = feature_vector[:6]
        
        return np.array(feature_vector, dtype=float)
    
    def _normalize_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        对特征向量进行归一化。
        如果有预训练的scaler，使用它进行转换；否则进行拟合。
        """
        try:
            if feature_vector.ndim == 1:
                feature_vector = feature_vector.reshape(1, -1)
            
            if self._use_pretrained_scaler:
                # 使用预训练的scaler直接转换
                normalized_vector = self.scaler.transform(feature_vector)
                self.logger.debug("使用预训练scaler进行特征归一化")
            else:
                # 训练模式：检查缩放器是否已经拟合过
                if hasattr(self.scaler, 'mean_'):
                    # 如果已拟合，只进行转换
                    normalized_vector = self.scaler.transform(feature_vector)
                else:
                    # 如果未拟合（第一次调用），则进行拟合并转换
                    self.logger.info("首次运行，正在使用当前数据作为基准拟合标准化缩放器...")
                    normalized_vector = self.scaler.fit_transform(feature_vector)
            
            return normalized_vector.flatten()
            
        except Exception as e:
            self.logger.error(f"特征归一化失败: {e}")
            return np.array([])
    
    def _update_history(self, data: Dict[str, float]):
        """
        更新历史数据窗口
        
        Args:
            data (Dict[str, float]): 当前数据
        """
        # 添加时间戳
        timestamped_data = {**data, 'timestamp': datetime.now().isoformat()}
        
        # 添加到历史数据
        self.historical_data.append(timestamped_data)
        
        # 保持窗口大小
        if len(self.historical_data) > self.history_window:
            self.historical_data.pop(0)
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            List[str]: 特征名称
        """
        return self.feature_names.copy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性（基于方差）
        
        Returns:
            Dict[str, float]: 特征重要性分数
        """
        importance = {}
        
        if len(self.historical_data) > 1:
            # 创建特征矩阵
            feature_matrix = []
            for data in self.historical_data:
                features = self._extract_basic_features(data)
                vector = self._convert_to_vector(features)
                feature_matrix.append(vector)
            
            feature_matrix = np.array(feature_matrix)
            
            # 计算每个特征的方差作为重要性指标
            if feature_matrix.shape[0] > 1:
                variances = np.var(feature_matrix, axis=0)
                
                for i, name in enumerate(self.feature_names):
                    if i < len(variances):
                        importance[name] = float(variances[i])
        
        return importance
    
    def reset_scaler(self):
        """重置归一化器"""
        self.scaler = StandardScaler()
        self.logger.info("特征归一化器已重置")
    
    def save_feature_config(self, filepath: str):
        """
        保存特征配置
        
        Args:
            filepath (str): 配置文件路径
        """
        try:
            config = {
                'feature_names': self.feature_names,
                'metrics_config': self.metrics_config,
                'history_window': self.history_window
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"特征配置已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"特征配置保存失败: {e}")
    
    def load_feature_config(self, filepath: str):
        """
        加载特征配置
        
        Args:
            filepath (str): 配置文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.feature_names = config.get('feature_names', [])
            self.history_window = config.get('history_window', 10)
            
            self.logger.info(f"特征配置已从 {filepath} 加载")
            
        except Exception as e:
            self.logger.warning(f"特征配置加载失败: {e}")


def create_feature_extractor(config: Dict, logger, scaler_path: Optional[str] = None) -> FeatureExtractor:
    """
    创建特征提取器实例
    
    Args:
        config (Dict): 配置参数
        logger: 日志记录器
        scaler_path: 预训练scaler的路径
        
    Returns:
        FeatureExtractor: 特征提取器实例
    """
    metrics = config.get('metrics', [])
    return FeatureExtractor(metrics, logger, scaler_path) 