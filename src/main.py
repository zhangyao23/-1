#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI网络异常检测系统 - 主程序入口

本程序是AI驱动的网络异常检测系统的主入口点。
系统能够实时监控网络状态，使用机器学习模型检测各种网络异常，
并在问题发生时自动保存关键数据用于后续分析。

主要功能：
1. 初始化各个子系统模块
2. 启动数据采集和监控循环
3. 协调异常检测和处理流程
4. 管理系统生命周期
"""

import os
import sys
import time
import signal
import threading
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


class NetworkAnomalyDetectorApp:
    """
    AI网络异常检测应用主类
    
    负责协调所有子系统，实现完整的异常检测工作流：
    1. 数据采集 -> 2. 特征提取 -> 3. AI推理 -> 4. 异常处理
    """
    
    def __init__(self, config_path="config/system_config.json"):
        """
        初始化异常检测应用
        
        Args:
            config_path (str): 系统配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.threads = []
        
        # 初始化日志系统
        self.logger = SystemLogger(self.config['logging'])
        self.logger.info("正在初始化AI网络异常检测系统...")
        
        # 初始化各个子系统
        self._initialize_components()
        
        # 设置信号处理器，用于优雅关闭
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self):
        """加载系统配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            sys.exit(1)
    
    def _initialize_components(self):
        """初始化所有系统组件"""
        try:
            # 初始化数据采集器
            self.data_collector = NetworkDataCollector(
                self.config['data_collection'], 
                self.logger
            )
            
            # 初始化特征提取器
            self.feature_extractor = FeatureExtractor(
                self.config['data_collection']['metrics'],
                self.logger
            )
            
            # 初始化环形缓冲区
            self.buffer_manager = CircularBuffer(
                self.config['buffer_manager'],
                self.logger
            )
            
            # 初始化AI模型
            self.autoencoder = AutoencoderModel(
                self.config['ai_models']['autoencoder'],
                self.logger
            )
            
            self.error_classifier = ErrorClassifier(
                self.config['ai_models']['classifier'],
                self.logger
            )
            
            # 初始化异常检测引擎
            self.anomaly_engine = AnomalyDetectionEngine(
                self.config['anomaly_detection'],
                self.autoencoder,
                self.error_classifier,
                self.buffer_manager,
                self.logger
            )
            
            self.logger.info("所有系统组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """信号处理器，用于优雅关闭系统"""
        self.logger.info(f"接收到信号 {signum}，正在关闭系统...")
        self.stop()
    
    def start(self):
        """启动异常检测系统"""
        try:
            self.logger.info("启动AI网络异常检测系统")
            self.running = True
            
            # 创建并启动数据采集线程
            data_collection_thread = threading.Thread(
                target=self._data_collection_loop,
                name="DataCollection"
            )
            data_collection_thread.daemon = True
            data_collection_thread.start()
            self.threads.append(data_collection_thread)
            
            # 创建并启动异常检测线程
            anomaly_detection_thread = threading.Thread(
                target=self._anomaly_detection_loop,
                name="AnomalyDetection"
            )
            anomaly_detection_thread.daemon = True
            anomaly_detection_thread.start()
            self.threads.append(anomaly_detection_thread)
            
            self.logger.info("系统启动成功，开始监控网络状态")
            
            # 主线程保持运行
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"系统启动失败: {e}")
            self.stop()
            raise
    
    def _data_collection_loop(self):
        """数据采集循环"""
        collection_interval = self.config['data_collection']['collection_interval']
        
        while self.running:
            try:
                # 采集网络数据
                raw_data = self.data_collector.collect_network_data()
                
                if raw_data:
                    # 提取特征
                    features = self.feature_extractor.extract_features(raw_data)
                    
                    # 添加到缓冲区
                    self.buffer_manager.add_data(features)
                    
                    self.logger.debug(f"采集数据成功，特征维度: {len(features)}")
                
                time.sleep(collection_interval)
                
            except Exception as e:
                self.logger.error(f"数据采集循环出错: {e}")
                time.sleep(collection_interval)
    
    def _anomaly_detection_loop(self):
        """异常检测循环"""
        detection_window = self.config['anomaly_detection']['detection_window']
        
        while self.running:
            try:
                # 获取最新数据进行异常检测
                recent_data = self.buffer_manager.get_recent_data(detection_window)
                
                if len(recent_data) >= detection_window:
                    # 执行异常检测
                    anomaly_result = self.anomaly_engine.detect_anomaly(recent_data)
                    
                    if anomaly_result['is_anomaly']:
                        self.logger.warning(
                            f"检测到异常: {anomaly_result['anomaly_type']}, "
                            f"置信度: {anomaly_result['confidence']:.3f}"
                        )
                        
                        # 处理异常事件
                        self._handle_anomaly(anomaly_result)
                
                time.sleep(detection_window)
                
            except Exception as e:
                self.logger.error(f"异常检测循环出错: {e}")
                time.sleep(detection_window)
    
    def _handle_anomaly(self, anomaly_result):
        """处理检测到的异常"""
        try:
            # 保存异常数据
            if self.config['anomaly_detection']['auto_save_enabled']:
                self.buffer_manager.save_anomaly_data(anomaly_result)
            
            # 发送通知
            if self.config['anomaly_detection']['notification_enabled']:
                self._send_notification(anomaly_result)
                
        except Exception as e:
            self.logger.error(f"异常处理失败: {e}")
    
    def _send_notification(self, anomaly_result):
        """发送异常通知"""
        notification_msg = (
            f"网络异常警报：检测到{anomaly_result['anomaly_type']}，"
            f"置信度：{anomaly_result['confidence']:.3f}，"
            f"时间：{anomaly_result['timestamp']}"
        )
        self.logger.warning(f"通知: {notification_msg}")
    
    def stop(self):
        """停止异常检测系统"""
        self.logger.info("正在停止系统...")
        self.running = False
        
        # 等待所有线程结束
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("系统已停止")
    
    def get_system_status(self):
        """获取系统状态信息"""
        status = {
            'running': self.running,
            'threads': len([t for t in self.threads if t.is_alive()]),
            'buffer_size': len(self.buffer_manager),
            'config_version': self.config['system']['version']
        }
        return status


def main():
    """主函数入口"""
    try:
        # 创建必要的目录
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/anomalies", exist_ok=True)
        os.makedirs("data/models", exist_ok=True)
        
        # 启动异常检测应用
        app = NetworkAnomalyDetectorApp()
        app.start()
        
    except KeyboardInterrupt:
        print("\n用户中断，系统正在退出...")
    except Exception as e:
        print(f"系统运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 