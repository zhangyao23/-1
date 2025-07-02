#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自编码器异常检测模型

实现基于自编码器的无监督异常检测算法。
自编码器通过学习正常数据的特征表示，当输入异常数据时
会产生较大的重构误差，从而实现异常检测。

主要功能：
1. 构建深度自编码器网络
2. 模型训练和验证
3. 异常检测推理
4. 重构误差计算和阈值判断
5. 模型保存和加载
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class AutoencoderModel:
    """
    自编码器异常检测模型
    
    基于深度自编码器的网络异常检测模型，
    通过重构误差来判断输入数据是否为异常。
    """
    
    def __init__(self, config: Dict, logger):
        """
        初始化自编码器模型
        
        Args:
            config (Dict): 模型配置参数
            logger: 系统日志记录器
        """
        self.config = config
        self.logger = logger
        
        # 模型参数
        self.input_features = config.get('input_features', 20)
        self.encoding_dim = config.get('encoding_dim', 10)
        self.threshold = config.get('threshold', 0.02)
        self.batch_size = config.get('batch_size', 32)
        # 切换到 TensorFlow SavedModel 目录格式
        self.model_path = config.get('model_path', 'models/autoencoder_model')
        
        # 模型组件
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        
        # SavedModel 相关属性
        self.saved_model = None
        self.model_inference = None
        
        # 训练历史
        self.training_history = {}
        self.threshold_history = []
        
        # 性能统计
        self.inference_count = 0
        self.anomaly_count = 0
        
        # 加载预训练模型（如果存在）
        self._load_model()
        
        # 检查是否有优化后的阈值
        self._load_optimized_threshold()
        
        self.logger.info("自编码器模型初始化完成")
    
    def build_model(self, input_dim: int = None) -> keras.Model:
        """
        构建自编码器模型
        
        Args:
            input_dim (int): 输入维度，如果为None则使用配置中的值
            
        Returns:
            keras.Model: 构建的自编码器模型
        """
        try:
            if input_dim is not None:
                self.input_features = input_dim
            
            # 输入层
            input_layer = keras.Input(shape=(self.input_features,), name='input')
            
            # 编码器部分
            # 第一层：降维
            encoded = layers.Dense(
                self.input_features // 2, 
                activation='relu', 
                name='encoder_layer1'
            )(input_layer)
            encoded = layers.Dropout(0.2)(encoded)
            
            # 第二层：进一步降维
            encoded = layers.Dense(
                self.encoding_dim, 
                activation='relu', 
                name='encoder_layer2'
            )(encoded)
            
            # 解码器部分
            # 第一层：开始升维
            decoded = layers.Dense(
                self.input_features // 2, 
                activation='relu', 
                name='decoder_layer1'
            )(encoded)
            decoded = layers.Dropout(0.2)(decoded)
            
            # 输出层：恢复原始维度
            decoded = layers.Dense(
                self.input_features, 
                activation='linear', 
                name='decoder_output'
            )(decoded)
            
            # 创建完整的自编码器模型
            self.model = keras.Model(input_layer, decoded, name='autoencoder')
            
            # 创建编码器子模型
            self.encoder = keras.Model(input_layer, encoded, name='encoder')
            
            # 编译模型
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info(f"自编码器模型构建完成，输入维度: {self.input_features}, 编码维度: {self.encoding_dim}")
            return self.model
            
        except Exception as e:
            self.logger.error(f"构建自编码器模型失败: {e}")
            raise
    
    def train(self, training_data: np.ndarray, epochs: int = 100, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        训练自编码器模型
        
        Args:
            training_data (np.ndarray): 训练数据（仅正常数据）
            epochs (int): 训练轮数
            validation_split (float): 验证集比例
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        try:
            self.logger.info(f"开始训练自编码器，数据形状: {training_data.shape}")
            
            # 动态设置输入特征数量
            actual_features = training_data.shape[1]
            if actual_features != self.input_features:
                self.logger.info(f"数据特征数量({actual_features})与配置不符({self.input_features})，自动调整配置")
                self.input_features = actual_features
                # 相应调整编码维度
                self.encoding_dim = max(min(self.input_features // 2, self.encoding_dim), 5)
                self.logger.info(f"调整后的编码维度: {self.encoding_dim}")
            
            # 数据预处理
            training_data = self._preprocess_data(training_data)
            
            # 构建模型（如果尚未构建或特征数量发生变化）
            if self.model is None:
                self.build_model(training_data.shape[1])
            
            # 设置回调函数
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # 训练模型
            history = self.model.fit(
                training_data, training_data,  # 自编码器：输入=输出
                epochs=epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # 保存训练历史
            self.training_history = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae']
            }
            
            # 计算重构误差阈值
            self._calculate_threshold(training_data)
            
            # 保存模型
            self._save_model()
            
            training_result = {
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'epochs_trained': len(history.history['loss']),
                'threshold': self.threshold,
                'input_features': self.input_features,
                'encoding_dim': self.encoding_dim,
                'model_saved': True
            }
            
            self.logger.info(f"模型训练完成，最终损失: {training_result['final_loss']:.6f}")
            return training_result
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        使用模型进行异常检测
        
        Args:
            input_data (np.ndarray): 输入数据
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        try:
            if self.model is None or self.model_inference is None:
                raise ValueError("模型尚未加载或训练")
            
            self.inference_count += 1
            
            # 数据预处理
            processed_data = self._preprocess_data(input_data)
            
            if processed_data.ndim == 1:
                processed_data = processed_data.reshape(1, -1)
            
            # 模型推理 - 使用SavedModel的推理函数
            input_tensor = tf.constant(processed_data, dtype=tf.float32)
            outputs = self.model_inference(inputs=input_tensor)
            reconstructed = outputs[list(outputs.keys())[0]].numpy()
            
            # 计算重构误差
            reconstruction_error = np.mean(np.square(processed_data - reconstructed), axis=1)
            
            # 判断是否为异常
            is_anomaly = reconstruction_error > self.threshold
            
            # 计算置信度（基于误差与阈值的比值）
            confidence = np.minimum(reconstruction_error / self.threshold, 2.0)
            
            if np.any(is_anomaly):
                self.anomaly_count += 1
            
            result = {
                'is_anomaly': bool(np.any(is_anomaly)),
                'reconstruction_error': float(reconstruction_error[0]) if len(reconstruction_error) == 1 else reconstruction_error.tolist(),
                'threshold': self.threshold,
                'confidence': float(confidence[0]) if len(confidence) == 1 else confidence.tolist(),
                'anomaly_score': float(reconstruction_error[0] / self.threshold) if len(reconstruction_error) == 1 else (reconstruction_error / self.threshold).tolist()
            }
            
            self.logger.debug(f"异常检测完成，重构误差: {result['reconstruction_error']:.6f}")
            return result
            
        except Exception as e:
            self.logger.error(f"异常检测预测失败: {e}")
            return {
                'is_anomaly': False,
                'reconstruction_error': 0.0,
                'threshold': self.threshold,
                'confidence': 0.0,
                'anomaly_score': 0.0,
                'error': str(e)
            }
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        数据预处理
        
        Args:
            data (np.ndarray): 原始数据
            
        Returns:
            np.ndarray: 预处理后的数据
        """
        try:
            # 处理缺失值和无穷值
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 确保数据维度正确
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # 标准化处理
            if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                # 使用已拟合的scaler
                data = self.scaler.transform(data)
            else:
                # 首次使用，拟合并转换
                data = self.scaler.fit_transform(data)
            
            return data.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            raise
    
    def _calculate_threshold(self, training_data: np.ndarray):
        """
        基于训练数据计算异常检测阈值
        
        Args:
            training_data (np.ndarray): 训练数据
        """
        try:
            # 使用训练数据计算重构误差分布
            if hasattr(self, 'model_inference') and self.model_inference is not None:
                # 使用SavedModel推理函数
                input_tensor = tf.constant(training_data, dtype=tf.float32)
                outputs = self.model_inference(inputs=input_tensor)
                reconstructed = outputs[list(outputs.keys())[0]].numpy()
            else:
                # 使用Keras模型
                reconstructed = self.model.predict(training_data, verbose=0)
            reconstruction_errors = np.mean(np.square(training_data - reconstructed), axis=1)
            
            # 使用95%分位数作为阈值
            self.threshold = np.percentile(reconstruction_errors, 95)
            
            # 确保阈值不会太小
            min_threshold = self.config.get('min_threshold', 0.001)
            self.threshold = max(self.threshold, min_threshold)
            
            # 记录阈值历史
            self.threshold_history.append({
                'threshold': self.threshold,
                'timestamp': datetime.now().isoformat(),
                'data_size': len(training_data)
            })
            
            self.logger.info(f"异常检测阈值已设置: {self.threshold:.6f}")
            
        except Exception as e:
            self.logger.error(f"计算阈值失败: {e}")
            # 使用默认阈值
            self.threshold = self.config.get('threshold', 0.02)
    
    def update_threshold(self, new_threshold: float):
        """
        更新异常检测阈值
        
        Args:
            new_threshold (float): 新的阈值
        """
        try:
            old_threshold = self.threshold
            self.threshold = new_threshold
            
            self.threshold_history.append({
                'threshold': self.threshold,
                'timestamp': datetime.now().isoformat(),
                'previous_threshold': old_threshold,
                'manual_update': True
            })
            
            self.logger.info(f"阈值已更新: {old_threshold:.6f} -> {new_threshold:.6f}")
            
        except Exception as e:
            self.logger.error(f"更新阈值失败: {e}")
    
    def get_encoding(self, input_data: np.ndarray) -> np.ndarray:
        """
        获取输入数据的编码表示
        
        Args:
            input_data (np.ndarray): 输入数据
            
        Returns:
            np.ndarray: 编码后的特征向量
        """
        try:
            if self.encoder is None:
                raise ValueError("编码器尚未构建")
            
            # 数据预处理
            processed_data = self._preprocess_data(input_data)
            
            # 获取编码
            encoding = self.encoder.predict(processed_data, verbose=0)
            
            return encoding
            
        except Exception as e:
            self.logger.error(f"获取编码失败: {e}")
            return np.array([])
    
    def _save_model(self):
        """保存模型到文件"""
        try:
            # 确保模型将要保存到的目录本身存在
            os.makedirs(self.model_path, exist_ok=True)
            
            # 使用 TensorFlow SavedModel 格式保存，不依赖 h5py
            tf.saved_model.save(self.model, self.model_path)
            
            # 保存scaler
            scaler_path = os.path.join(self.model_path, 'autoencoder_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
            # 保存模型配置和统计信息
            config_path = os.path.join(self.model_path, 'autoencoder_config.json')
            model_info = {
                'config': self.config,
                'threshold': float(self.threshold),
                'threshold_history': self.threshold_history,
                'training_history': self.training_history,
                'input_features': self.input_features,
                'encoding_dim': self.encoding_dim,
                'inference_count': self.inference_count,
                'anomaly_count': self.anomaly_count,
                'saved_at': datetime.now().isoformat()
            }
            
            import json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"模型已保存到: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
    
    def _load_model(self):
        """加载预训练的自编码器模型和scaler"""
        try:
            model_dir = self.model_path
            scaler_file = os.path.join(self.model_path, 'autoencoder_scaler.pkl')
            config_file = os.path.join(self.model_path, 'autoencoder_config.json')
            
            if os.path.exists(model_dir):
                # 使用 TensorFlow SavedModel 格式加载
                self.saved_model = tf.saved_model.load(model_dir)
                
                # 获取推理函数
                self.model_inference = self.saved_model.signatures['serving_default']
                self.logger.info(f"成功从 {model_dir} 加载预训练模型")
                
                # 加载保存的配置和阈值
                if os.path.exists(config_file):
                    import json
                    with open(config_file, 'r', encoding='utf-8') as f:
                        saved_config = json.load(f)
                    
                    # 使用保存的阈值
                    saved_threshold = saved_config.get('threshold', self.threshold)
                    if saved_threshold != self.threshold:
                        self.logger.info(f"使用保存的阈值: {saved_threshold:.6f} (配置文件阈值: {self.threshold:.6f})")
                        self.threshold = saved_threshold
                    
                    # 恢复其他保存的信息
                    self.threshold_history = saved_config.get('threshold_history', [])
                    self.training_history = saved_config.get('training_history', {})
                    self.inference_count = saved_config.get('inference_count', 0)
                    self.anomaly_count = saved_config.get('anomaly_count', 0)
                    
                    # 更新模型参数
                    self.input_features = saved_config.get('input_features', self.input_features)
                    self.encoding_dim = saved_config.get('encoding_dim', self.encoding_dim)
                    
                    self.logger.info(f"成功加载模型配置，阈值: {self.threshold:.6f}")
                else:
                    self.logger.warning(f"未找到配置文件 {config_file}，使用默认阈值")
                
                if os.path.exists(scaler_file):
                    # 检查scaler的特征数量是否与当前数据匹配
                    try:
                        with open(scaler_file, 'rb') as f:
                            temp_scaler = joblib.load(f)
                        
                        # 检查scaler的特征数量
                        if hasattr(temp_scaler, 'n_features_in_'):
                            saved_features = temp_scaler.n_features_in_
                            if saved_features != self.input_features:
                                self.logger.warning(f"保存的scaler特征数量({saved_features})与当前配置({self.input_features})不匹配，将重新初始化scaler")
                                # 重新初始化scaler
                                self.scaler = StandardScaler()
                            else:
                                self.scaler = temp_scaler
                                self.logger.info(f"成功从 {scaler_file} 加载预训练的scaler")
                        else:
                            self.scaler = temp_scaler
                            self.logger.info(f"成功从 {scaler_file} 加载预训练的scaler")
                    except Exception as scaler_error:
                        self.logger.warning(f"加载scaler失败: {scaler_error}，将使用新的scaler")
                        self.scaler = StandardScaler()
                else:
                    self.logger.warning(f"未找到对应的scaler文件 {scaler_file}，将使用新的scaler")
                
                # 标记模型已加载
                self.model = self.saved_model
                
            else:
                self.logger.warning(f"模型文件 {model_dir} 不存在，需要先训练模型")
                self.model = None
                self.model_inference = None

        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            self.logger.info("将清理旧模型文件并重新开始训练")
            # 清理可能损坏的模型文件
            if os.path.exists(self.model_path):
                import shutil
                try:
                    shutil.rmtree(self.model_path)
                    self.logger.info(f"已清理损坏的模型文件: {self.model_path}")
                except Exception as cleanup_error:
                    self.logger.warning(f"清理模型文件失败: {cleanup_error}")
            
            self.model = None
            self.model_inference = None
            self.scaler = StandardScaler()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        summary = {
            'model_loaded': self.model is not None,
            'input_features': self.input_features,
            'encoding_dim': self.encoding_dim,
            'threshold': self.threshold,
            'inference_count': self.inference_count,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(self.inference_count, 1) * 100
        }
        
        if self.model:
            summary['model_parameters'] = self.model.count_params()
            summary['model_layers'] = len(self.model.layers)
        
        if self.training_history:
            summary['training_epochs'] = len(self.training_history.get('loss', []))
            summary['final_loss'] = self.training_history.get('loss', [0])[-1]
        
        return summary
    
    def evaluate_performance(self, test_data: np.ndarray, true_labels: np.ndarray = None) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            test_data (np.ndarray): 测试数据
            true_labels (np.ndarray): 真实标签（可选）
            
        Returns:
            Dict[str, Any]: 性能评估结果
        """
        try:
            # 预测
            predictions = self.predict(test_data)
            reconstruction_errors = np.array([predictions['reconstruction_error']])
            
            performance = {
                'test_samples': len(test_data),
                'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
                'std_reconstruction_error': float(np.std(reconstruction_errors)),
                'max_reconstruction_error': float(np.max(reconstruction_errors)),
                'threshold': self.threshold
            }
            
            # 如果有真实标签，计算分类指标
            if true_labels is not None:
                predicted_labels = reconstruction_errors > self.threshold
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                performance.update({
                    'accuracy': accuracy_score(true_labels, predicted_labels),
                    'precision': precision_score(true_labels, predicted_labels, zero_division=0),
                    'recall': recall_score(true_labels, predicted_labels, zero_division=0),
                    'f1_score': f1_score(true_labels, predicted_labels, zero_division=0)
                })
            
            return performance
            
        except Exception as e:
            self.logger.error(f"性能评估失败: {e}")
            return {}
    
    def _load_optimized_threshold(self):
        """加载优化后的阈值（如果存在）"""
        try:
            threshold_file = os.path.join(self.model_path, 'optimized_threshold.txt')
            if os.path.exists(threshold_file):
                with open(threshold_file, 'r') as f:
                    optimized_threshold = float(f.read().strip())
                old_threshold = self.threshold
                self.threshold = optimized_threshold
                self.logger.info(f"使用优化后的阈值: {optimized_threshold:.6f} (原阈值: {old_threshold:.6f})")
        except Exception as e:
            self.logger.warning(f"加载优化阈值失败: {e}")
    
    def save_optimized_threshold(self, threshold: float):
        """保存优化后的阈值"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            threshold_file = os.path.join(self.model_path, 'optimized_threshold.txt')
            with open(threshold_file, 'w') as f:
                f.write(str(threshold))
            self.logger.info(f"优化阈值已保存: {threshold:.6f}")
        except Exception as e:
            self.logger.error(f"保存优化阈值失败: {e}")


def create_autoencoder_model(config: Dict, logger) -> AutoencoderModel:
    """
    创建自编码器模型实例
    
    Args:
        config (Dict): 配置参数
        logger: 日志记录器
        
    Returns:
        AutoencoderModel: 自编码器模型实例
    """
    return AutoencoderModel(config, logger) 