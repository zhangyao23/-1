#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
环形缓冲区管理模块

实现高效的环形缓冲区数据结构，用于存储网络监控数据。
当检测到异常时，能够快速保存缓冲区中的历史数据。

主要功能：
1. 固定大小的环形缓冲区实现
2. 线程安全的数据操作
3. 数据压缩和存储优化
4. 异常事件数据快速保存
5. 缓冲区状态监控和管理
"""

import time
import json
import threading
import gzip
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import deque
from pathlib import Path


class CircularBuffer:
    """
    环形缓冲区管理器
    
    使用环形缓冲区存储最近的网络监控数据，
    当检测到异常时能够快速保存相关的历史数据作为证据。
    """
    
    def __init__(self, config: Dict, logger):
        """
        初始化环形缓冲区
        
        Args:
            config (Dict): 缓冲区配置参数
            logger: 系统日志记录器
        """
        self.config = config
        self.logger = logger
        
        # 缓冲区配置
        self.buffer_size = config.get('buffer_size', 1000)
        self.data_retention_minutes = config.get('data_retention_minutes', 30)
        self.save_threshold = config.get('save_threshold', 0.8)
        self.compression_enabled = config.get('compression_enabled', True)
        
        # 缓冲区数据结构
        self.buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.RLock()
        
        # 统计信息
        self.total_items_added = 0
        self.total_items_removed = 0
        self.last_save_time = None
        
        # 存储路径
        self.anomaly_data_path = config.get('anomaly_data_path', 'data/anomalies/')
        Path(self.anomaly_data_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"环形缓冲区初始化完成，容量: {self.buffer_size}")
    
    def add_data(self, data: Dict[str, Any]) -> bool:
        """
        向缓冲区添加数据
        
        Args:
            data (Dict[str, Any]): 要添加的数据
            
        Returns:
            bool: 添加是否成功
        """
        try:
            with self.buffer_lock:
                # 添加时间戳
                timestamped_data = {
                    **data,
                    'buffer_timestamp': time.time(),
                    'buffer_datetime': datetime.now().isoformat()
                }
                
                # 如果缓冲区已满，自动移除最旧的数据
                if len(self.buffer) >= self.buffer_size:
                    removed_data = self.buffer.popleft()
                    self.total_items_removed += 1
                    self.logger.debug(f"缓冲区已满，移除最旧数据: {removed_data.get('buffer_datetime', 'Unknown')}")
                
                # 添加新数据
                self.buffer.append(timestamped_data)
                self.total_items_added += 1
                
                self.logger.debug(f"数据已添加到缓冲区，当前大小: {len(self.buffer)}")
                return True
                
        except Exception as e:
            self.logger.error(f"向缓冲区添加数据失败: {e}")
            return False
    
    def get_recent_data(self, count: int) -> List[Dict[str, Any]]:
        """
        获取最近的N条数据
        
        Args:
            count (int): 要获取的数据条数
            
        Returns:
            List[Dict[str, Any]]: 最近的数据列表
        """
        try:
            with self.buffer_lock:
                if count <= 0:
                    return []
                
                # 获取最后count条数据
                recent_data = list(self.buffer)[-count:]
                
                self.logger.debug(f"获取最近 {len(recent_data)} 条数据")
                return recent_data
                
        except Exception as e:
            self.logger.error(f"获取最近数据失败: {e}")
            return []
    
    def get_data_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        根据时间范围获取数据
        
        Args:
            start_time (datetime): 开始时间
            end_time (datetime): 结束时间
            
        Returns:
            List[Dict[str, Any]]: 时间范围内的数据
        """
        try:
            with self.buffer_lock:
                filtered_data = []
                
                start_timestamp = start_time.timestamp()
                end_timestamp = end_time.timestamp()
                
                for data in self.buffer:
                    data_timestamp = data.get('buffer_timestamp', 0)
                    if start_timestamp <= data_timestamp <= end_timestamp:
                        filtered_data.append(data)
                
                self.logger.debug(f"时间范围 {start_time} - {end_time} 内找到 {len(filtered_data)} 条数据")
                return filtered_data
                
        except Exception as e:
            self.logger.error(f"按时间范围获取数据失败: {e}")
            return []
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """
        获取缓冲区中的所有数据
        
        Returns:
            List[Dict[str, Any]]: 所有数据
        """
        try:
            with self.buffer_lock:
                all_data = list(self.buffer)
                self.logger.debug(f"获取缓冲区所有数据，共 {len(all_data)} 条")
                return all_data
                
        except Exception as e:
            self.logger.error(f"获取所有数据失败: {e}")
            return []
    
    def save_anomaly_data(self, anomaly_info: Dict[str, Any]) -> Optional[str]:
        """
        保存异常相关的缓冲区数据
        
        Args:
            anomaly_info (Dict[str, Any]): 异常信息
            
        Returns:
            Optional[str]: 保存的文件路径，失败返回None
        """
        try:
            # 生成保存文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            anomaly_type = anomaly_info.get('anomaly_type', 'unknown')
            filename = f"anomaly_{anomaly_type}_{timestamp}.json"
            
            if self.compression_enabled:
                filename += ".gz"
            
            filepath = os.path.join(self.anomaly_data_path, filename)
            
            # 准备要保存的数据
            save_data = {
                'anomaly_info': anomaly_info,
                'buffer_metadata': self.get_buffer_stats(),
                'saved_at': datetime.now().isoformat(),
                'buffer_data': self.get_all_data()
            }
            
            # 保存数据
            if self.compression_enabled:
                self._save_compressed_data(save_data, filepath)
            else:
                self._save_json_data(save_data, filepath)
            
            self.last_save_time = datetime.now()
            self.logger.info(f"异常数据已保存到: {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"保存异常数据失败: {e}")
            return None
    
    def _save_json_data(self, data: Dict[str, Any], filepath: str):
        """
        保存JSON格式数据
        
        Args:
            data (Dict[str, Any]): 要保存的数据
            filepath (str): 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def _save_compressed_data(self, data: Dict[str, Any], filepath: str):
        """
        保存压缩格式数据
        
        Args:
            data (Dict[str, Any]): 要保存的数据
            filepath (str): 文件路径
        """
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        json_bytes = json_str.encode('utf-8')
        
        with gzip.open(filepath, 'wb') as f:
            f.write(json_bytes)
    
    def load_anomaly_data(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        加载已保存的异常数据
        
        Args:
            filepath (str): 文件路径
            
        Returns:
            Optional[Dict[str, Any]]: 加载的数据，失败返回None
        """
        try:
            if filepath.endswith('.gz'):
                return self._load_compressed_data(filepath)
            else:
                return self._load_json_data(filepath)
                
        except Exception as e:
            self.logger.error(f"加载异常数据失败: {e}")
            return None
    
    def _load_json_data(self, filepath: str) -> Dict[str, Any]:
        """
        加载JSON格式数据
        
        Args:
            filepath (str): 文件路径
            
        Returns:
            Dict[str, Any]: 加载的数据
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_compressed_data(self, filepath: str) -> Dict[str, Any]:
        """
        加载压缩格式数据
        
        Args:
            filepath (str): 文件路径
            
        Returns:
            Dict[str, Any]: 加载的数据
        """
        with gzip.open(filepath, 'rb') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode('utf-8')
            return json.loads(json_str)
    
    def cleanup_old_data(self):
        """清理缓冲区中的过期数据"""
        try:
            current_time = time.time()
            retention_seconds = self.data_retention_minutes * 60
            
            with self.buffer_lock:
                # 从左侧（最旧数据）开始检查
                while self.buffer:
                    oldest_data = self.buffer[0]
                    data_timestamp = oldest_data.get('buffer_timestamp', current_time)
                    
                    if current_time - data_timestamp > retention_seconds:
                        removed_data = self.buffer.popleft()
                        self.total_items_removed += 1
                        self.logger.debug(f"清理过期数据: {removed_data.get('buffer_datetime', 'Unknown')}")
                    else:
                        # 由于deque是有序的，一旦找到未过期数据就可以停止
                        break
                
                self.logger.debug(f"数据清理完成，当前缓冲区大小: {len(self.buffer)}")
                
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {e}")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        获取缓冲区统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            with self.buffer_lock:
                stats = {
                    'current_size': len(self.buffer),
                    'max_size': self.buffer_size,
                    'usage_percentage': (len(self.buffer) / self.buffer_size) * 100,
                    'total_items_added': self.total_items_added,
                    'total_items_removed': self.total_items_removed,
                    'retention_minutes': self.data_retention_minutes,
                    'compression_enabled': self.compression_enabled,
                    'last_save_time': self.last_save_time.isoformat() if self.last_save_time else None
                }
                
                # 计算数据时间范围
                if self.buffer:
                    oldest_timestamp = self.buffer[0].get('buffer_timestamp', 0)
                    newest_timestamp = self.buffer[-1].get('buffer_timestamp', 0)
                    
                    stats['data_time_range'] = {
                        'oldest': datetime.fromtimestamp(oldest_timestamp).isoformat(),
                        'newest': datetime.fromtimestamp(newest_timestamp).isoformat(),
                        'span_minutes': (newest_timestamp - oldest_timestamp) / 60
                    }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"获取缓冲区统计失败: {e}")
            return {}
    
    def is_nearly_full(self) -> bool:
        """
        检查缓冲区是否接近满载
        
        Returns:
            bool: 是否接近满载
        """
        try:
            with self.buffer_lock:
                usage_rate = len(self.buffer) / self.buffer_size
                return usage_rate >= self.save_threshold
                
        except Exception as e:
            self.logger.error(f"检查缓冲区状态失败: {e}")
            return False
    
    def clear_buffer(self):
        """清空缓冲区"""
        try:
            with self.buffer_lock:
                cleared_count = len(self.buffer)
                self.buffer.clear()
                self.logger.info(f"缓冲区已清空，清理了 {cleared_count} 条数据")
                
        except Exception as e:
            self.logger.error(f"清空缓冲区失败: {e}")
    
    def resize_buffer(self, new_size: int):
        """
        调整缓冲区大小
        
        Args:
            new_size (int): 新的缓冲区大小
        """
        try:
            with self.buffer_lock:
                if new_size <= 0:
                    self.logger.warning("缓冲区大小必须大于0")
                    return
                
                old_size = self.buffer_size
                self.buffer_size = new_size
                
                # 创建新的deque并复制数据
                old_data = list(self.buffer)
                self.buffer = deque(old_data, maxlen=new_size)
                
                self.logger.info(f"缓冲区大小已调整: {old_size} -> {new_size}")
                
        except Exception as e:
            self.logger.error(f"调整缓冲区大小失败: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        估算缓冲区内存使用情况
        
        Returns:
            Dict[str, Any]: 内存使用信息
        """
        try:
            import sys
            
            with self.buffer_lock:
                total_size = 0
                item_count = len(self.buffer)
                
                # 估算每个数据项的大小
                if self.buffer:
                    sample_item = self.buffer[0]
                    item_size = sys.getsizeof(sample_item)
                    
                    # 递归计算嵌套对象大小
                    for value in sample_item.values():
                        item_size += sys.getsizeof(value)
                    
                    total_size = item_size * item_count
                
                memory_info = {
                    'total_memory_bytes': total_size,
                    'total_memory_mb': total_size / (1024 * 1024),
                    'avg_item_size_bytes': total_size / item_count if item_count > 0 else 0,
                    'item_count': item_count
                }
                
                return memory_info
                
        except Exception as e:
            self.logger.error(f"计算内存使用失败: {e}")
            return {}
    
    def export_to_csv(self, filepath: str, time_range: Optional[tuple] = None) -> bool:
        """
        导出缓冲区数据到CSV文件
        
        Args:
            filepath (str): CSV文件路径
            time_range (Optional[tuple]): 时间范围 (start_time, end_time)
            
        Returns:
            bool: 导出是否成功
        """
        try:
            import pandas as pd
            
            # 获取要导出的数据
            if time_range:
                start_time, end_time = time_range
                data_list = self.get_data_by_time_range(start_time, end_time)
            else:
                data_list = self.get_all_data()
            
            if not data_list:
                self.logger.warning("没有数据可导出")
                return False
            
            # 转换为DataFrame
            df = pd.DataFrame(data_list)
            
            # 保存到CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            self.logger.info(f"数据已导出到CSV: {filepath}，共 {len(data_list)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"导出CSV失败: {e}")
            return False
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        with self.buffer_lock:
            return len(self.buffer)
    
    def __contains__(self, item: Dict[str, Any]) -> bool:
        """检查缓冲区是否包含某个数据项"""
        with self.buffer_lock:
            return item in self.buffer
    
    def __str__(self) -> str:
        """返回缓冲区的字符串表示"""
        stats = self.get_buffer_stats()
        return (f"CircularBuffer(size={stats.get('current_size', 0)}/"
                f"{stats.get('max_size', 0)}, "
                f"usage={stats.get('usage_percentage', 0):.1f}%)")


def create_circular_buffer(config: Dict, logger) -> CircularBuffer:
    """
    创建环形缓冲区实例
    
    Args:
        config (Dict): 配置参数
        logger: 日志记录器
        
    Returns:
        CircularBuffer: 环形缓冲区实例
    """
    return CircularBuffer(config, logger) 