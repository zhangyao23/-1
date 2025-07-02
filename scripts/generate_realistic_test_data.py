#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真实网络错误场景测试数据生成器

基于实际网络环境中常见的问题，生成各种真实的错误场景测试数据：
1. WiFi信号问题
2. 网络拥塞
3. 硬件故障
4. 配置错误
5. 攻击和安全问题
6. 环境干扰
7. 系统资源问题

使用方法：
python scripts/generate_realistic_test_data.py
"""

import json
import numpy as np
import random
from typing import Dict, List, Any

class RealisticTestDataGenerator:
    """真实网络错误场景数据生成器"""
    
    def __init__(self):
        # 正常基准值
        self.normal_baseline = {
            'wlan0_wireless_quality': 70.0,
            'wlan0_wireless_level': -55.0,
            'wlan0_packet_loss_rate': 0.01,
            'wlan0_send_rate_bps': 500000.0,
            'wlan0_recv_rate_bps': 1500000.0,
            'tcp_retrans_segments': 5,
            'gateway_ping_time': 12.5,
            'dns_response_time': 25.0,
            'tcp_connection_count': 30,
            'cpu_percent': 15.0,
            'memory_percent': 45.0
        }
        
        self.test_scenarios = []
    
    def generate_all_scenarios(self) -> List[Dict]:
        """生成所有真实错误场景"""
        
        # 1. WiFi信号相关问题
        self._generate_wifi_signal_issues()
        
        # 2. 网络拥塞问题
        self._generate_network_congestion()
        
        # 3. 硬件故障问题
        self._generate_hardware_failures()
        
        # 4. 网络配置错误
        self._generate_configuration_errors()
        
        # 5. 安全攻击场景
        self._generate_security_attacks()
        
        # 6. 环境干扰问题
        self._generate_environmental_interference()
        
        # 7. 系统资源问题
        self._generate_system_resource_issues()
        
        # 8. 边缘网络场景
        self._generate_edge_network_scenarios()
        
        # 9. 移动网络问题
        self._generate_mobile_network_issues()
        
        # 10. 正常参考场景
        self._generate_normal_scenarios()
        
        return self.test_scenarios
    
    def _generate_wifi_signal_issues(self):
        """生成WiFi信号相关问题"""
        
        # 1. 信号衰减（距离路由器过远）
        scenario = self._create_base_scenario("WiFi信号衰减", "用户离路由器太远，信号质量下降")
        scenario['data'].update({
            'wlan0_wireless_quality': 25.0,  # 信号质量很差
            'wlan0_wireless_level': -85.0,   # 信号强度很弱
            'wlan0_packet_loss_rate': 0.12,  # 丢包率增加
            'gateway_ping_time': 45.0,       # 延迟增加
            'wlan0_send_rate_bps': 150000.0, # 发送速率下降
            'wlan0_recv_rate_bps': 300000.0  # 接收速率下降
        })
        scenario['expected_type'] = 'signal_degradation'
        self.test_scenarios.append(scenario)
        
        # 2. 信号间歇性中断
        scenario = self._create_base_scenario("WiFi信号间歇中断", "信号不稳定，间歇性中断")
        scenario['data'].update({
            'wlan0_wireless_quality': 35.0,
            'wlan0_wireless_level': -78.0,
            'wlan0_packet_loss_rate': 0.25,  # 高丢包率
            'tcp_retrans_segments': 45,      # 重传次数增加
            'gateway_ping_time': 120.0,      # 延迟波动大
            'dns_response_time': 180.0
        })
        scenario['expected_type'] = 'intermittent_connectivity'
        self.test_scenarios.append(scenario)
        
        # 3. 信号完全丢失
        scenario = self._create_base_scenario("WiFi信号丢失", "完全失去WiFi连接")
        scenario['data'].update({
            'wlan0_wireless_quality': 0.0,
            'wlan0_wireless_level': -95.0,
            'wlan0_packet_loss_rate': 1.0,   # 100%丢包
            'wlan0_send_rate_bps': 0.0,
            'wlan0_recv_rate_bps': 0.0,
            'gateway_ping_time': 5000.0,     # 超时
            'dns_response_time': 5000.0,
            'tcp_connection_count': 0
        })
        scenario['expected_type'] = 'connection_lost'
        self.test_scenarios.append(scenario)
    
    def _generate_network_congestion(self):
        """生成网络拥塞问题"""
        
        # 1. 带宽饱和
        scenario = self._create_base_scenario("带宽饱和", "网络带宽被大量使用占满")
        scenario['data'].update({
            'wlan0_send_rate_bps': 50000.0,   # 发送速率下降
            'wlan0_recv_rate_bps': 100000.0,  # 接收速率下降
            'gateway_ping_time': 200.0,       # 延迟增加
            'dns_response_time': 300.0,
            'tcp_connection_count': 150,      # 连接数增加
            'cpu_percent': 45.0,              # CPU使用率上升
            'wlan0_packet_loss_rate': 0.08
        })
        scenario['expected_type'] = 'bandwidth_saturation'
        self.test_scenarios.append(scenario)
        
        # 2. 网络队列拥塞
        scenario = self._create_base_scenario("队列拥塞", "网络设备队列满，数据包排队")
        scenario['data'].update({
            'gateway_ping_time': 350.0,
            'dns_response_time': 450.0,
            'wlan0_packet_loss_rate': 0.15,
            'tcp_retrans_segments': 80,
            'tcp_connection_count': 200,
            'memory_percent': 75.0
        })
        scenario['expected_type'] = 'queue_congestion'
        self.test_scenarios.append(scenario)
        
        # 3. P2P流量影响
        scenario = self._create_base_scenario("P2P流量干扰", "P2P下载占用大量带宽")
        scenario['data'].update({
            'wlan0_recv_rate_bps': 80000.0,
            'wlan0_send_rate_bps': 2000000.0,  # 上传流量大
            'gateway_ping_time': 180.0,
            'tcp_connection_count': 500,       # 连接数激增
            'cpu_percent': 65.0,
            'memory_percent': 80.0
        })
        scenario['expected_type'] = 'p2p_interference'
        self.test_scenarios.append(scenario)
    
    def _generate_hardware_failures(self):
        """生成硬件故障问题"""
        
        # 1. 网卡驱动问题
        scenario = self._create_base_scenario("网卡驱动故障", "网络适配器驱动程序异常")
        scenario['data'].update({
            'wlan0_packet_loss_rate': 0.45,
            'tcp_retrans_segments': 120,
            'wlan0_send_rate_bps': 80000.0,
            'wlan0_recv_rate_bps': 200000.0,
            'gateway_ping_time': 800.0,
            'cpu_percent': 85.0,              # CPU异常高
            'tcp_connection_count': 8         # 连接数异常低
        })
        scenario['expected_type'] = 'driver_malfunction'
        self.test_scenarios.append(scenario)
        
        # 2. 路由器过热
        scenario = self._create_base_scenario("路由器过热", "路由器因过热性能下降")
        scenario['data'].update({
            'wlan0_wireless_quality': 45.0,
            'gateway_ping_time': 250.0,
            'dns_response_time': 400.0,
            'wlan0_packet_loss_rate': 0.18,
            'wlan0_send_rate_bps': 200000.0,
            'wlan0_recv_rate_bps': 400000.0,
            'tcp_retrans_segments': 60
        })
        scenario['expected_type'] = 'hardware_overheating'
        self.test_scenarios.append(scenario)
        
        # 3. 内存泄漏
        scenario = self._create_base_scenario("系统内存泄漏", "应用程序内存泄漏影响网络")
        scenario['data'].update({
            'memory_percent': 95.0,           # 内存使用率极高
            'cpu_percent': 78.0,
            'gateway_ping_time': 450.0,
            'dns_response_time': 600.0,
            'tcp_connection_count': 5,        # 连接数异常低
            'wlan0_packet_loss_rate': 0.22
        })
        scenario['expected_type'] = 'memory_leak'
        self.test_scenarios.append(scenario)
    
    def _generate_configuration_errors(self):
        """生成网络配置错误"""
        
        # 1. DNS配置错误
        scenario = self._create_base_scenario("DNS配置错误", "DNS服务器配置不当")
        scenario['data'].update({
            'dns_response_time': 8000.0,      # DNS超时
            'gateway_ping_time': 15.0,        # 网关正常
            'tcp_connection_count': 12,       # 连接数下降
            'wlan0_wireless_quality': 65.0,   # 信号正常
            'wlan0_packet_loss_rate': 0.02
        })
        scenario['expected_type'] = 'dns_misconfiguration'
        self.test_scenarios.append(scenario)
        
        # 2. IP地址冲突
        scenario = self._create_base_scenario("IP地址冲突", "网络中存在IP地址冲突")
        scenario['data'].update({
            'wlan0_packet_loss_rate': 0.35,
            'tcp_retrans_segments': 100,
            'gateway_ping_time': 2000.0,      # 间歇性超时
            'tcp_connection_count': 3,
            'wlan0_send_rate_bps': 50000.0,
            'wlan0_recv_rate_bps': 80000.0
        })
        scenario['expected_type'] = 'ip_conflict'
        self.test_scenarios.append(scenario)
        
        # 3. 防火墙阻塞
        scenario = self._create_base_scenario("防火墙过度阻塞", "防火墙规则过于严格")
        scenario['data'].update({
            'tcp_connection_count': 8,        # 连接被阻塞
            'gateway_ping_time': 15.0,        # 基本连接正常
            'dns_response_time': 30.0,
            'wlan0_send_rate_bps': 100000.0,  # 发送受限
            'wlan0_recv_rate_bps': 1200000.0, # 接收正常
            'cpu_percent': 25.0
        })
        scenario['expected_type'] = 'firewall_blocking'
        self.test_scenarios.append(scenario)
    
    def _generate_security_attacks(self):
        """生成安全攻击场景"""
        
        # 1. DDoS攻击
        scenario = self._create_base_scenario("DDoS攻击", "遭受分布式拒绝服务攻击")
        scenario['data'].update({
            'tcp_connection_count': 2000,     # 连接数激增
            'cpu_percent': 98.0,              # CPU满负荷
            'memory_percent': 92.0,
            'gateway_ping_time': 3000.0,
            'wlan0_packet_loss_rate': 0.65,
            'tcp_retrans_segments': 500,
            'wlan0_send_rate_bps': 10000.0,
            'wlan0_recv_rate_bps': 50000.0
        })
        scenario['expected_type'] = 'ddos_attack'
        self.test_scenarios.append(scenario)
        
        # 2. ARP欺骗攻击
        scenario = self._create_base_scenario("ARP欺骗攻击", "网络中存在ARP欺骗")
        scenario['data'].update({
            'gateway_ping_time': 800.0,
            'wlan0_packet_loss_rate': 0.28,
            'tcp_retrans_segments': 90,
            'dns_response_time': 1500.0,
            'wlan0_send_rate_bps': 150000.0,
            'wlan0_recv_rate_bps': 300000.0,
            'tcp_connection_count': 18
        })
        scenario['expected_type'] = 'arp_spoofing'
        self.test_scenarios.append(scenario)
        
        # 3. 恶意软件网络活动
        scenario = self._create_base_scenario("恶意软件活动", "系统感染恶意软件")
        scenario['data'].update({
            'cpu_percent': 88.0,
            'memory_percent': 85.0,
            'tcp_connection_count': 800,      # 异常多的连接
            'wlan0_send_rate_bps': 1800000.0, # 异常上传流量
            'gateway_ping_time': 160.0,
            'dns_response_time': 250.0,
            'wlan0_packet_loss_rate': 0.12
        })
        scenario['expected_type'] = 'malware_activity'
        self.test_scenarios.append(scenario)
    
    def _generate_environmental_interference(self):
        """生成环境干扰问题"""
        
        # 1. 2.4GHz频段干扰
        scenario = self._create_base_scenario("2.4GHz频段干扰", "微波炉等设备干扰WiFi")
        scenario['data'].update({
            'wlan0_wireless_quality': 30.0,
            'wlan0_wireless_level': -70.0,
            'wlan0_packet_loss_rate': 0.20,
            'gateway_ping_time': 150.0,
            'tcp_retrans_segments': 55,
            'wlan0_send_rate_bps': 180000.0,
            'wlan0_recv_rate_bps': 350000.0
        })
        scenario['expected_type'] = 'frequency_interference'
        self.test_scenarios.append(scenario)
        
        # 2. 邻居WiFi干扰
        scenario = self._create_base_scenario("邻居WiFi干扰", "周围WiFi网络过多造成干扰")
        scenario['data'].update({
            'wlan0_wireless_quality': 40.0,
            'wlan0_wireless_level': -65.0,
            'wlan0_packet_loss_rate': 0.08,
            'gateway_ping_time': 85.0,
            'dns_response_time': 120.0,
            'tcp_retrans_segments': 25,
            'wlan0_send_rate_bps': 250000.0,
            'wlan0_recv_rate_bps': 600000.0
        })
        scenario['expected_type'] = 'neighbor_interference'
        self.test_scenarios.append(scenario)
        
        # 3. 电磁干扰
        scenario = self._create_base_scenario("电磁干扰", "附近电子设备产生电磁干扰")
        scenario['data'].update({
            'wlan0_wireless_quality': 20.0,
            'wlan0_wireless_level': -82.0,
            'wlan0_packet_loss_rate': 0.35,
            'gateway_ping_time': 300.0,
            'tcp_retrans_segments': 120,
            'wlan0_send_rate_bps': 80000.0,
            'wlan0_recv_rate_bps': 150000.0,
            'tcp_connection_count': 8
        })
        scenario['expected_type'] = 'electromagnetic_interference'
        self.test_scenarios.append(scenario)
    
    def _generate_system_resource_issues(self):
        """生成系统资源问题"""
        
        # 1. CPU过载
        scenario = self._create_base_scenario("CPU过载", "CPU使用率过高影响网络性能")
        scenario['data'].update({
            'cpu_percent': 95.0,
            'gateway_ping_time': 180.0,
            'dns_response_time': 250.0,
            'tcp_connection_count': 15,
            'wlan0_packet_loss_rate': 0.06,
            'memory_percent': 70.0
        })
        scenario['expected_type'] = 'cpu_overload'
        self.test_scenarios.append(scenario)
        
        # 2. 磁盘I/O瓶颈
        scenario = self._create_base_scenario("磁盘I/O瓶颈", "磁盘读写过慢影响系统")
        scenario['data'].update({
            'cpu_percent': 25.0,              # CPU正常
            'memory_percent': 88.0,           # 内存高
            'gateway_ping_time': 120.0,
            'dns_response_time': 200.0,
            'tcp_connection_count': 12,
            'wlan0_packet_loss_rate': 0.04
        })
        scenario['expected_type'] = 'disk_io_bottleneck'
        self.test_scenarios.append(scenario)
        
        # 3. 网络缓冲区溢出
        scenario = self._create_base_scenario("网络缓冲区溢出", "网络缓冲区满导致丢包")
        scenario['data'].update({
            'wlan0_packet_loss_rate': 0.40,
            'tcp_retrans_segments': 150,
            'gateway_ping_time': 400.0,
            'cpu_percent': 60.0,
            'memory_percent': 90.0,
            'tcp_connection_count': 45
        })
        scenario['expected_type'] = 'buffer_overflow'
        self.test_scenarios.append(scenario)
    
    def _generate_edge_network_scenarios(self):
        """生成边缘网络场景"""
        
        # 1. ISP网络问题
        scenario = self._create_base_scenario("ISP网络故障", "互联网服务提供商网络问题")
        scenario['data'].update({
            'gateway_ping_time': 15.0,        # 本地网关正常
            'dns_response_time': 5000.0,      # 外部DNS超时
            'wlan0_wireless_quality': 75.0,   # WiFi正常
            'wlan0_packet_loss_rate': 0.02,
            'tcp_connection_count': 8,        # 外部连接失败
            'wlan0_send_rate_bps': 400000.0,
            'wlan0_recv_rate_bps': 100000.0   # 下载受限
        })
        scenario['expected_type'] = 'isp_network_issue'
        self.test_scenarios.append(scenario)
        
        # 2. 骨干网拥塞
        scenario = self._create_base_scenario("骨干网拥塞", "互联网骨干网络拥塞")
        scenario['data'].update({
            'gateway_ping_time': 12.0,
            'dns_response_time': 800.0,
            'wlan0_send_rate_bps': 450000.0,
            'wlan0_recv_rate_bps': 200000.0,
            'tcp_connection_count': 20,
            'wlan0_packet_loss_rate': 0.03
        })
        scenario['expected_type'] = 'backbone_congestion'
        self.test_scenarios.append(scenario)
    
    def _generate_mobile_network_issues(self):
        """生成移动网络问题"""
        
        # 1. 频繁漫游
        scenario = self._create_base_scenario("频繁网络漫游", "设备在不同接入点间频繁切换")
        scenario['data'].update({
            'wlan0_wireless_quality': 55.0,
            'wlan0_wireless_level': -68.0,
            'gateway_ping_time': 80.0,
            'wlan0_packet_loss_rate': 0.12,
            'tcp_retrans_segments': 35,
            'tcp_connection_count': 25,
            'dns_response_time': 60.0
        })
        scenario['expected_type'] = 'frequent_roaming'
        self.test_scenarios.append(scenario)
        
        # 2. 接入点故障
        scenario = self._create_base_scenario("接入点故障", "WiFi接入点硬件故障")
        scenario['data'].update({
            'wlan0_wireless_quality': 15.0,
            'wlan0_wireless_level': -88.0,
            'wlan0_packet_loss_rate': 0.60,
            'gateway_ping_time': 2500.0,
            'tcp_retrans_segments': 200,
            'wlan0_send_rate_bps': 20000.0,
            'wlan0_recv_rate_bps': 30000.0,
            'tcp_connection_count': 2
        })
        scenario['expected_type'] = 'access_point_failure'
        self.test_scenarios.append(scenario)
    
    def _generate_normal_scenarios(self):
        """生成正常参考场景"""
        
        # 1. 标准正常状态
        scenario = self._create_base_scenario("正常网络状态", "一切正常的网络环境")
        scenario['data'] = self.normal_baseline.copy()
        scenario['expected_type'] = 'normal'
        self.test_scenarios.append(scenario)
        
        # 2. 轻负载状态
        scenario = self._create_base_scenario("轻负载状态", "网络使用量较低的正常状态")
        scenario['data'].update({
            'wlan0_wireless_quality': 85.0,
            'wlan0_wireless_level': -45.0,
            'wlan0_packet_loss_rate': 0.005,
            'gateway_ping_time': 8.0,
            'dns_response_time': 15.0,
            'tcp_connection_count': 12,
            'cpu_percent': 8.0,
            'memory_percent': 30.0
        })
        scenario['expected_type'] = 'normal_light_load'
        self.test_scenarios.append(scenario)
        
        # 3. 高负载但正常状态
        scenario = self._create_base_scenario("高负载正常状态", "网络使用量高但仍在正常范围")
        scenario['data'].update({
            'wlan0_send_rate_bps': 800000.0,
            'wlan0_recv_rate_bps': 2000000.0,
            'tcp_connection_count': 80,
            'cpu_percent': 45.0,
            'memory_percent': 65.0,
            'gateway_ping_time': 25.0,
            'dns_response_time': 40.0
        })
        scenario['expected_type'] = 'normal_high_load'
        self.test_scenarios.append(scenario)
    
    def _create_base_scenario(self, name: str, description: str) -> Dict:
        """创建基础场景模板"""
        return {
            'name': name,
            'description': description,
            'data': self.normal_baseline.copy(),
            'expected_anomaly': True,
            'expected_type': 'unknown',
            'severity': 'medium',
            'timestamp': '2025-07-02T10:00:00Z'
        }
    
    def save_scenarios_to_file(self, filepath: str = 'data/realistic_test_scenarios.json'):
        """保存场景到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.test_scenarios, f, ensure_ascii=False, indent=2)
            print(f"✅ 已生成 {len(self.test_scenarios)} 个真实测试场景")
            print(f"📁 保存到: {filepath}")
            return True
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False
    
    def print_scenario_summary(self):
        """打印场景总结"""
        print("\n" + "="*60)
        print("🔍 真实网络错误场景测试数据生成完成")
        print("="*60)
        
        # 按类型统计
        type_counts = {}
        for scenario in self.test_scenarios:
            expected_type = scenario.get('expected_type', 'unknown')
            type_counts[expected_type] = type_counts.get(expected_type, 0) + 1
        
        print(f"\n📊 场景统计 (总共 {len(self.test_scenarios)} 个):")
        for scenario_type, count in sorted(type_counts.items()):
            print(f"  {scenario_type}: {count} 个")
        
        print(f"\n📋 场景详情:")
        for i, scenario in enumerate(self.test_scenarios, 1):
            print(f"  {i:2d}. {scenario['name']}")
            print(f"      {scenario['description']}")
            print(f"      类型: {scenario['expected_type']}")

def main():
    """主函数"""
    print("🚀 开始生成真实网络错误场景测试数据...")
    
    generator = RealisticTestDataGenerator()
    scenarios = generator.generate_all_scenarios()
    
    # 保存到文件
    success = generator.save_scenarios_to_file()
    
    if success:
        # 打印总结
        generator.print_scenario_summary()
        
        print(f"\n💡 使用方法:")
        print(f"   python scripts/test_with_real_data.py --file data/realistic_test_scenarios.json")
        print(f"   或使用交互式测试加载具体场景数据")
    
    return scenarios

if __name__ == "__main__":
    main() 