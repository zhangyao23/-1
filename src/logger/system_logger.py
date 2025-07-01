#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统日志管理模块

提供统一的日志记录功能，支持：
1. 多级别日志记录（DEBUG, INFO, WARNING, ERROR, CRITICAL）
2. 日志文件自动轮转和归档
3. 控制台和文件双重输出
4. 线程安全的日志记录
5. 自定义日志格式和过滤器
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


class SystemLogger:
    """
    系统日志管理器
    
    统一管理整个异常检测系统的日志记录，确保日志信息的完整性和可追溯性。
    支持多种输出目标和灵活的配置选项。
    """
    
    def __init__(self, config):
        """
        初始化日志系统
        
        Args:
            config (dict): 日志配置参数，包含级别、文件路径、格式等设置
        """
        self.config = config
        self.logger = logging.getLogger('AnomalyDetector')
        self.logger.setLevel(getattr(logging, config.get('level', 'INFO')))
        
        # 清除已有的处理器，避免重复日志
        self.logger.handlers.clear()
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建日志目录
        log_dir = Path(config['file_path']).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 添加文件处理器
        self._add_file_handler()
        
        # 添加控制台处理器
        if config.get('console_output', True):
            self._add_console_handler()
        
        # 记录初始化信息
        self.info("系统日志模块初始化完成")
    
    def _add_file_handler(self):
        """添加文件日志处理器，支持自动轮转"""
        try:
            # 解析文件大小配置
            max_bytes = self._parse_file_size(self.config.get('max_file_size', '10MB'))
            backup_count = self.config.get('backup_count', 5)
            
            # 创建轮转文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.config['file_path'],
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            file_handler.setFormatter(self.formatter)
            file_handler.setLevel(getattr(logging, self.config.get('level', 'INFO')))
            
            self.logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"文件日志处理器创建失败: {e}")
    
    def _add_console_handler(self):
        """添加控制台日志处理器"""
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            
            # 控制台通常显示INFO及以上级别的日志
            console_level = self.config.get('console_level', 'INFO')
            console_handler.setLevel(getattr(logging, console_level))
            
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"控制台日志处理器创建失败: {e}")
    
    def _parse_file_size(self, size_str):
        """
        解析文件大小字符串
        
        Args:
            size_str (str): 文件大小字符串，如 '10MB', '1GB'
            
        Returns:
            int: 字节数
        """
        size_str = size_str.upper()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def debug(self, message, *args, **kwargs):
        """记录DEBUG级别日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        """记录INFO级别日志"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """记录WARNING级别日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """记录ERROR级别日志"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        """记录CRITICAL级别日志"""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message, *args, **kwargs):
        """记录异常信息，会自动包含堆栈跟踪"""
        self.logger.exception(message, *args, **kwargs)
    
    def log_system_metrics(self, metrics):
        """
        记录系统性能指标
        
        Args:
            metrics (dict): 系统指标数据
        """
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.info(f"系统指标 - {metrics_str}")
    
    def log_anomaly_event(self, anomaly_data):
        """
        记录异常事件详细信息
        
        Args:
            anomaly_data (dict): 异常事件数据
        """
        self.warning(
            f"异常事件 - 类型: {anomaly_data.get('type', 'Unknown')}, "
            f"置信度: {anomaly_data.get('confidence', 0):.3f}, "
            f"时间: {anomaly_data.get('timestamp', 'Unknown')}"
        )
    
    def log_model_performance(self, model_name, performance_data):
        """
        记录AI模型性能数据
        
        Args:
            model_name (str): 模型名称
            performance_data (dict): 性能数据
        """
        perf_str = ", ".join([f"{k}: {v}" for k, v in performance_data.items()])
        self.info(f"模型性能 [{model_name}] - {perf_str}")
    
    def create_session_log(self, session_id):
        """
        为特定会话创建专用日志记录器
        
        Args:
            session_id (str): 会话标识符
            
        Returns:
            logging.Logger: 会话专用日志记录器
        """
        session_logger = logging.getLogger(f'AnomalyDetector.Session.{session_id}')
        session_logger.setLevel(self.logger.level)
        
        # 复用现有的处理器
        for handler in self.logger.handlers:
            session_logger.addHandler(handler)
        
        return session_logger
    
    def get_log_stats(self):
        """
        获取日志统计信息
        
        Returns:
            dict: 日志统计数据
        """
        stats = {
            'log_level': self.config.get('level', 'INFO'),
            'handlers_count': len(self.logger.handlers),
            'log_file': self.config.get('file_path', 'N/A'),
            'console_output': self.config.get('console_output', True)
        }
        
        # 尝试获取日志文件大小
        try:
            log_file_path = Path(self.config['file_path'])
            if log_file_path.exists():
                stats['log_file_size'] = log_file_path.stat().st_size
            else:
                stats['log_file_size'] = 0
        except Exception:
            stats['log_file_size'] = 'Unknown'
        
        return stats
    
    def set_log_level(self, level):
        """
        动态调整日志级别
        
        Args:
            level (str): 新的日志级别
        """
        try:
            new_level = getattr(logging, level.upper())
            self.logger.setLevel(new_level)
            self.info(f"日志级别已调整为: {level.upper()}")
        except AttributeError:
            self.error(f"无效的日志级别: {level}")
    
    def flush_logs(self):
        """强制刷新所有日志处理器"""
        for handler in self.logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
    
    def close_handlers(self):
        """关闭所有日志处理器"""
        self.info("正在关闭日志处理器...")
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()


class AnomalyEventLogger:
    """
    异常事件专用日志记录器
    
    为异常检测事件提供结构化的日志记录功能，
    便于后续的数据分析和模式识别。
    """
    
    def __init__(self, system_logger, event_log_path="logs/anomaly_events.log"):
        """
        初始化异常事件日志记录器
        
        Args:
            system_logger (SystemLogger): 系统日志记录器实例
            event_log_path (str): 异常事件日志文件路径
        """
        self.system_logger = system_logger
        self.event_log_path = event_log_path
        
        # 创建事件日志目录
        Path(event_log_path).parent.mkdir(parents=True, exist_ok=True)
    
    def log_detection_event(self, event_data):
        """
        记录异常检测事件
        
        Args:
            event_data (dict): 包含异常详细信息的字典
        """
        timestamp = datetime.now().isoformat()
        event_record = {
            'timestamp': timestamp,
            'event_type': 'anomaly_detection',
            'data': event_data
        }
        
        # 记录到系统日志
        self.system_logger.log_anomaly_event(event_data)
        
        # 记录到专用事件日志文件
        self._write_event_log(event_record)
    
    def log_model_inference(self, model_name, input_data, output_data, inference_time):
        """
        记录模型推理过程
        
        Args:
            model_name (str): 模型名称
            input_data: 输入数据
            output_data: 输出结果
            inference_time (float): 推理耗时（秒）
        """
        event_record = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'model_inference',
            'model_name': model_name,
            'inference_time': inference_time,
            'input_shape': getattr(input_data, 'shape', 'N/A'),
            'output_summary': str(output_data)[:100] if output_data else 'None'
        }
        
        self._write_event_log(event_record)
    
    def _write_event_log(self, event_record):
        """
        将事件记录写入日志文件
        
        Args:
            event_record (dict): 事件记录数据
        """
        try:
            import json
            with open(self.event_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_record, ensure_ascii=False) + '\n')
        except Exception as e:
            self.system_logger.error(f"写入事件日志失败: {e}")


def get_logger(name="AnomalyDetector"):
    """
    获取日志记录器实例
    
    Args:
        name (str): 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器实例
    """
    return logging.getLogger(name) 