#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网络数据采集模块

负责从操作系统和网络接口收集各种网络统计数据，包括：
1. WiFi信号强度和连接状态
2. 网络接口流量统计
3. 连接质量指标
4. 系统网络性能数据

支持多种网络接口和协议，提供统一的数据接口。
"""

import os
import time
import subprocess
import psutil
import netifaces
from datetime import datetime
from typing import Dict, List, Optional, Any


class NetworkDataCollector:
    """
    网络数据采集器
    
    从系统底层API和命令行工具收集网络相关的统计数据，
    为异常检测模型提供原始数据输入。
    """
    
    def __init__(self, config: Dict, logger):
        """
        初始化网络数据采集器
        
        Args:
            config (Dict): 数据采集配置参数
            logger: 系统日志记录器
        """
        self.config = config
        self.logger = logger
        self.interfaces = config.get('interfaces', ['wlan0', 'eth0'])
        self.collection_interval = config.get('collection_interval', 5)
        self.timeout = config.get('timeout', 10)
        self.metrics = config.get('metrics', [])
        
        # 初始化上一次的统计数据，用于计算差值
        self.previous_stats = {}
        
        self.logger.info(f"网络数据采集器初始化完成，监控接口: {self.interfaces}")
    
    def collect_network_data(self) -> Dict[str, Any]:
        """
        收集完整的网络统计数据
        
        Returns:
            Dict[str, Any]: 包含所有网络指标的字典
        """
        timestamp = datetime.now()
        collected_data = {
            'timestamp': timestamp.isoformat(),
            'collection_time': time.time()
        }
        
        try:
            # 收集各种网络指标
            if 'signal_strength' in self.metrics:
                collected_data.update(self._get_wifi_signal_strength())
            
            if 'packet_loss_rate' in self.metrics:
                collected_data.update(self._get_packet_loss_rate())
            
            if 'data_rate' in self.metrics:
                collected_data.update(self._get_data_rates())
            
            if 'retry_count' in self.metrics:
                collected_data.update(self._get_retry_counts())
            
            if 'latency' in self.metrics:
                collected_data.update(self._get_network_latency())
            
            if 'throughput' in self.metrics:
                collected_data.update(self._get_throughput_stats())
            
            if 'connection_count' in self.metrics:
                collected_data.update(self._get_connection_counts())
            
            if 'beacon_interval' in self.metrics:
                collected_data.update(self._get_beacon_intervals())
            
            # 收集系统级网络统计
            collected_data.update(self._get_system_network_stats())
            
            self.logger.debug(f"网络数据采集完成，指标数量: {len(collected_data)}")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"网络数据采集失败: {e}")
            return {}
    
    def _get_wifi_signal_strength(self) -> Dict[str, float]:
        """
        获取WiFi信号强度信息
        
        Returns:
            Dict[str, float]: 信号强度相关指标
        """
        signal_data = {}
        
        try:
            # 从/proc/net/wireless读取WiFi信号信息
            with open('/proc/net/wireless', 'r') as f:
                lines = f.readlines()
                
            for line in lines[2:]:  # 跳过头部行
                parts = line.split()
                if len(parts) >= 4:
                    interface = parts[0].rstrip(':')
                    if interface in self.interfaces:
                        signal_data[f'{interface}_wireless_quality'] = float(parts[2])
                        signal_data[f'{interface}_wireless_level'] = float(parts[3])
                        
        except Exception as e:
            self.logger.debug(f"WiFi信号强度获取失败: {e}")
        
        return signal_data
    
    def _get_packet_loss_rate(self) -> Dict[str, float]:
        """
        计算网络接口的丢包率
        
        Returns:
            Dict[str, float]: 各接口的丢包率
        """
        packet_loss_data = {}
        
        try:
            current_stats = {}
            
            for interface in self.interfaces:
                io_counters = psutil.net_io_counters(pernic=True).get(interface)
                if io_counters:
                    current_stats[interface] = {
                        'packets_sent': io_counters.packets_sent,
                        'packets_recv': io_counters.packets_recv,
                        'dropin': io_counters.dropin,
                        'dropout': io_counters.dropout
                    }
            
            # 计算与上次采集的差值
            if self.previous_stats.get('packet_stats'):
                for interface in current_stats:
                    if interface in self.previous_stats['packet_stats']:
                        prev = self.previous_stats['packet_stats'][interface]
                        curr = current_stats[interface]
                        
                        total_packets = (curr['packets_sent'] - prev['packets_sent']) + \
                                      (curr['packets_recv'] - prev['packets_recv'])
                        dropped_packets = (curr['dropin'] - prev['dropin']) + \
                                        (curr['dropout'] - prev['dropout'])
                        
                        if total_packets > 0:
                            loss_rate = (dropped_packets / total_packets) * 100
                            packet_loss_data[f'{interface}_packet_loss_rate'] = loss_rate
            
            self.previous_stats['packet_stats'] = current_stats
            
        except Exception as e:
            self.logger.warning(f"丢包率计算失败: {e}")
        
        return packet_loss_data
    
    def _get_data_rates(self) -> Dict[str, float]:
        """
        获取网络接口的数据传输速率
        
        Returns:
            Dict[str, float]: 数据传输速率指标
        """
        data_rate_info = {}
        
        try:
            current_time = time.time()
            current_stats = {}
            
            for interface in self.interfaces:
                io_counters = psutil.net_io_counters(pernic=True).get(interface)
                if io_counters:
                    current_stats[interface] = {
                        'bytes_sent': io_counters.bytes_sent,
                        'bytes_recv': io_counters.bytes_recv,
                        'timestamp': current_time
                    }
            
            # 计算传输速率
            if self.previous_stats.get('data_rate_stats'):
                for interface in current_stats:
                    if interface in self.previous_stats['data_rate_stats']:
                        prev = self.previous_stats['data_rate_stats'][interface]
                        curr = current_stats[interface]
                        
                        time_diff = curr['timestamp'] - prev['timestamp']
                        if time_diff > 0:
                            send_rate = (curr['bytes_sent'] - prev['bytes_sent']) / time_diff
                            recv_rate = (curr['bytes_recv'] - prev['bytes_recv']) / time_diff
                            
                            data_rate_info[f'{interface}_send_rate_bps'] = send_rate
                            data_rate_info[f'{interface}_recv_rate_bps'] = recv_rate
            
            self.previous_stats['data_rate_stats'] = current_stats
            
        except Exception as e:
            self.logger.warning(f"数据传输速率计算失败: {e}")
        
        return data_rate_info
    
    def _get_retry_counts(self) -> Dict[str, int]:
        """
        获取网络重传统计
        
        Returns:
            Dict[str, int]: 重传计数器
        """
        retry_data = {}
        
        try:
            # 尝试从/proc/net/snmp获取TCP重传统计
            with open('/proc/net/snmp', 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if line.startswith('Tcp:') and i + 1 < len(lines):
                    headers = line.split()[1:]
                    values = lines[i + 1].split()[1:]
                    
                    tcp_stats = dict(zip(headers, map(int, values)))
                    
                    # 提取重传相关指标
                    if 'RetransSegs' in tcp_stats:
                        retry_data['tcp_retrans_segments'] = tcp_stats['RetransSegs']
                    if 'InSegs' in tcp_stats:
                        retry_data['tcp_in_segments'] = tcp_stats['InSegs']
                    if 'OutSegs' in tcp_stats:
                        retry_data['tcp_out_segments'] = tcp_stats['OutSegs']
                    
                    break
            
        except Exception as e:
            self.logger.debug(f"重传统计获取失败: {e}")
        
        return retry_data
    
    def _get_network_latency(self) -> Dict[str, float]:
        """
        测量网络延迟
        
        Returns:
            Dict[str, float]: 网络延迟指标
        """
        latency_data = {}
        
        try:
            # 使用ping测试延迟
            test_hosts = ['8.8.8.8']
            
            for host in test_hosts:
                try:
                    result = subprocess.run(
                        ['ping', '-c', '1', '-W', '3', host],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0 and 'time=' in result.stdout:
                        # 简单解析延迟时间
                        output = result.stdout
                        start_idx = output.find('time=') + 5
                        end_idx = output.find(' ms', start_idx)
                        
                        if start_idx > 4 and end_idx > start_idx:
                            latency = float(output[start_idx:end_idx])
                            latency_data[f'ping_latency_{host.replace(".", "_")}'] = latency
                
                except Exception:
                    latency_data[f'ping_latency_{host.replace(".", "_")}'] = 999.0
            
        except Exception as e:
            self.logger.debug(f"网络延迟测量失败: {e}")
        
        return latency_data
    
    def _get_throughput_stats(self) -> Dict[str, float]:
        """
        获取网络吞吐量统计
        
        Returns:
            Dict[str, float]: 吞吐量相关指标
        """
        throughput_data = {}
        
        try:
            net_stats = psutil.net_io_counters(pernic=True)
            
            for interface in self.interfaces:
                if interface in net_stats:
                    stats = net_stats[interface]
                    throughput_data[f'{interface}_bytes_sent'] = stats.bytes_sent
                    throughput_data[f'{interface}_bytes_recv'] = stats.bytes_recv
                    throughput_data[f'{interface}_packets_sent'] = stats.packets_sent
                    throughput_data[f'{interface}_packets_recv'] = stats.packets_recv
            
        except Exception as e:
            self.logger.warning(f"吞吐量统计获取失败: {e}")
        
        return throughput_data
    
    def _get_connection_counts(self) -> Dict[str, int]:
        """
        获取网络连接统计
        
        Returns:
            Dict[str, int]: 连接数量统计
        """
        connection_data = {}
        
        try:
            connections = psutil.net_connections()
            
            # 按状态统计连接数
            status_counts = {}
            for conn in connections:
                status = conn.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            for status, count in status_counts.items():
                connection_data[f'connections_{status.lower()}'] = count
            
            connection_data['total_connections'] = len(connections)
            
        except Exception as e:
            self.logger.warning(f"连接统计获取失败: {e}")
        
        return connection_data
    
    def _get_beacon_intervals(self) -> Dict[str, float]:
        """
        获取WiFi beacon间隔信息
        
        Returns:
            Dict[str, float]: Beacon间隔指标
        """
        beacon_data = {}
        
        try:
            # 尝试使用iw命令获取详细WiFi信息
            for interface in self.interfaces:
                if 'wlan' in interface:
                    try:
                        result = subprocess.run(
                            ['iw', interface, 'scan'],
                            capture_output=True,
                            text=True,
                            timeout=self.timeout
                        )
                        
                        if result.returncode == 0:
                            output = result.stdout
                            
                            # 解析beacon间隔
                            if 'beacon interval:' in output.lower():
                                beacon_lines = [line for line in output.split('\n') 
                                              if 'beacon interval:' in line.lower()]
                                if beacon_lines:
                                    interval = self._extract_beacon_interval(beacon_lines[0])
                                    beacon_data[f'{interface}_beacon_interval'] = interval
                    
                    except subprocess.TimeoutExpired:
                        self.logger.debug(f"接口 {interface} beacon扫描超时")
                    except Exception as e:
                        self.logger.debug(f"接口 {interface} beacon获取失败: {e}")
            
        except Exception as e:
            self.logger.debug(f"Beacon间隔获取失败: {e}")
        
        return beacon_data
    
    def _extract_beacon_interval(self, beacon_line: str) -> float:
        """
        从iw scan输出中提取beacon间隔
        
        Args:
            beacon_line (str): 包含beacon间隔的行
            
        Returns:
            float: Beacon间隔(TU - Time Units)
        """
        try:
            # 查找beacon interval后面的数值
            parts = beacon_line.lower().split('beacon interval:')
            if len(parts) > 1:
                interval_part = parts[1].strip().split()[0]
                return float(interval_part)
        except Exception:
            pass
        
        return 100.0  # 默认beacon间隔
    
    def _get_system_network_stats(self) -> Dict[str, Any]:
        """
        获取系统级网络统计信息
        
        Returns:
            Dict[str, Any]: 系统网络指标
        """
        system_stats = {}
        
        try:
            # CPU和内存使用率
            system_stats['cpu_percent'] = psutil.cpu_percent(interval=1)
            system_stats['memory_percent'] = psutil.virtual_memory().percent
            
            # 网络接口状态
            net_if_stats = psutil.net_if_stats()
            for interface in self.interfaces:
                if interface in net_if_stats:
                    stats = net_if_stats[interface]
                    system_stats[f'{interface}_is_up'] = stats.isup
                    system_stats[f'{interface}_speed'] = stats.speed
            
        except Exception as e:
            self.logger.debug(f"系统网络统计获取失败: {e}")
        
        return system_stats
    
    def get_interface_info(self) -> Dict[str, Dict]:
        """
        获取网络接口详细信息
        
        Returns:
            Dict[str, Dict]: 每个接口的详细信息
        """
        interface_info = {}
        
        try:
            # 获取所有网络接口信息
            all_interfaces = netifaces.interfaces()
            
            for interface in self.interfaces:
                if interface in all_interfaces:
                    interface_data = {}
                    
                    # 获取接口地址信息
                    addrs = netifaces.ifaddresses(interface)
                    
                    # IPv4地址
                    if netifaces.AF_INET in addrs:
                        ipv4_info = addrs[netifaces.AF_INET][0]
                        interface_data['ipv4_addr'] = ipv4_info.get('addr', '')
                        interface_data['ipv4_netmask'] = ipv4_info.get('netmask', '')
                    
                    # 硬件地址
                    if netifaces.AF_LINK in addrs:
                        link_info = addrs[netifaces.AF_LINK][0]
                        interface_data['mac_addr'] = link_info.get('addr', '')
                    
                    interface_info[interface] = interface_data
            
        except Exception as e:
            self.logger.error(f"接口信息获取失败: {e}")
        
        return interface_info
    
    def validate_interfaces(self) -> bool:
        """
        验证配置的网络接口是否可用
        
        Returns:
            bool: 接口是否全部可用
        """
        try:
            available_interfaces = netifaces.interfaces()
            
            for interface in self.interfaces:
                if interface not in available_interfaces:
                    self.logger.warning(f"网络接口 {interface} 不可用")
                    return False
            
            self.logger.info("所有配置的网络接口验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"接口验证失败: {e}")
            return False
    
    def _get_dns_response_time(self) -> float:
        """
        获取DNS响应时间
        
        Returns:
            float: DNS响应时间(毫秒)
        """
        try:
            import socket
            import time
            
            start_time = time.time()
            socket.gethostbyname('google.com')
            end_time = time.time()
            
            return (end_time - start_time) * 1000  # 转换为毫秒
            
        except Exception as e:
            self.logger.debug(f"DNS响应时间获取失败: {e}")
            return 100.0  # 默认值
    
    def _get_gateway_ping_time(self) -> float:
        """
        获取网关ping时间
        
        Returns:
            float: 网关ping时间(毫秒)
        """
        try:
            import subprocess
            
            # 获取默认网关
            gateways = netifaces.gateways()
            default_gateway = gateways['default'][netifaces.AF_INET][0]
            
            # ping网关
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '3', default_gateway],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # 解析ping时间
                output = result.stdout
                if 'time=' in output:
                    time_part = output.split('time=')[1].split(' ')[0]
                    return float(time_part)
            
            return 100.0  # 默认值
            
        except Exception as e:
            self.logger.debug(f"网关ping时间获取失败: {e}")
            return 100.0
    
    def _get_packet_error_rate(self) -> float:
        """
        获取数据包错误率
        
        Returns:
            float: 数据包错误率(百分比)
        """
        try:
            net_stats = psutil.net_io_counters()
            
            total_packets = net_stats.packets_sent + net_stats.packets_recv
            error_packets = net_stats.errin + net_stats.errout + net_stats.dropin + net_stats.dropout
            
            if total_packets > 0:
                error_rate = (error_packets / total_packets) * 100
                return min(error_rate, 100.0)  # 限制在0-100%
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"数据包错误率获取失败: {e}")
            return 0.0
    
    def _get_bandwidth_utilization(self) -> float:
        """
        获取带宽利用率
        
        Returns:
            float: 带宽利用率(百分比)
        """
        try:
            import time
            
            # 获取当前网络IO统计
            net_stats1 = psutil.net_io_counters()
            time.sleep(1)
            net_stats2 = psutil.net_io_counters()
            
            # 计算每秒传输的字节数
            bytes_per_sec = (net_stats2.bytes_sent + net_stats2.bytes_recv) - \
                           (net_stats1.bytes_sent + net_stats1.bytes_recv)
            
            # 假设最大带宽为100Mbps (12.5MB/s)
            max_bandwidth = 12.5 * 1024 * 1024  # bytes/sec
            utilization = (bytes_per_sec / max_bandwidth) * 100
            
            return min(utilization, 100.0)
            
        except Exception as e:
            self.logger.debug(f"带宽利用率获取失败: {e}")
            return 0.0
    
    def _get_authentication_attempts(self, interface: str) -> int:
        """
        获取认证尝试次数
        
        Args:
            interface (str): 网络接口名
            
        Returns:
            int: 认证尝试次数
        """
        try:
            # 读取系统日志中的认证信息
            import subprocess
            
            result = subprocess.run(
                ['journalctl', '-u', 'wpa_supplicant', '--since', '1 hour ago', '-q'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                auth_lines = [line for line in result.stdout.split('\n') 
                             if 'authentication' in line.lower() and interface in line]
                return len(auth_lines)
            
            return 0
            
        except Exception as e:
            self.logger.debug(f"认证尝试次数获取失败 {interface}: {e}")
            return 0
    
    def _get_interference_level(self, interface: str) -> float:
        """
        获取信号干扰水平
        
        Args:
            interface (str): 网络接口名
            
        Returns:
            float: 干扰水平(0-100)
        """
        try:
            if 'wlan' not in interface:
                return 0.0
            
            # 尝试获取WiFi扫描结果中的信号强度分布
            result = subprocess.run(
                ['iw', interface, 'scan'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout
                signal_strengths = []
                
                for line in output.split('\n'):
                    if 'signal:' in line.lower():
                        try:
                            signal_str = line.split('signal:')[1].strip().split()[0]
                            signal_val = float(signal_str)
                            signal_strengths.append(signal_val)
                        except:
                            continue
                
                # 计算信号分布的方差作为干扰指标
                if len(signal_strengths) > 1:
                    import statistics
                    variance = statistics.variance(signal_strengths)
                    # 将方差映射到0-100的干扰水平
                    interference = min(variance / 10.0, 100.0)
                    return interference
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"干扰水平获取失败 {interface}: {e}")
            return 0.0


def create_network_collector(config: Dict, logger) -> NetworkDataCollector:
    """
    创建网络数据采集器实例
    
    Args:
        config (Dict): 配置信息
        logger: 日志记录器
        
    Returns:
        NetworkDataCollector: 数据采集器实例
    """
    return NetworkDataCollector(config, logger) 