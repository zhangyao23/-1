#!/usr/bin/env python3
"""
SNPE性能测试脚本
测试AI网络异常检测程序在SNPE环境中的性能表现
"""

import os
import sys
import subprocess
import json
import time
import statistics
from pathlib import Path

class SNPEPerformanceTester:
    def __init__(self):
        self.project_root = Path.cwd()
        self.executable = self.project_root / "dlc_mobile_inference"
        self.results = {"performance_tests": [], "summary": {}}
        
    def log_performance(self, test_name, avg_time_ms, min_time_ms, max_time_ms, 
                       std_dev_ms, throughput_fps, details=""):
        """记录性能测试结果"""
        result = {
            "test_name": test_name,
            "avg_time_ms": avg_time_ms,
            "min_time_ms": min_time_ms,
            "max_time_ms": max_time_ms,
            "std_dev_ms": std_dev_ms,
            "throughput_fps": throughput_fps,
            "details": details
        }
        self.results["performance_tests"].append(result)
        
        print(f"📊 {test_name}")
        print(f"   平均延迟: {avg_time_ms:.1f}ms")
        print(f"   延迟范围: {min_time_ms:.1f}-{max_time_ms:.1f}ms")
        print(f"   标准差: {std_dev_ms:.1f}ms")
        print(f"   吞吐量: {throughput_fps:.1f} FPS")
        if details:
            print(f"   详情: {details}")
        print()
        
    def run_inference_batch(self, test_name, input_file, num_runs=20):
        """运行批量推理测试"""
        if not self.executable.exists():
            print(f"❌ 可执行文件不存在: {self.executable}")
            return None
            
        input_path = self.project_root / input_file
        if not input_path.exists():
            print(f"❌ 测试数据不存在: {input_path}")
            return None
            
        print(f"🚀 开始性能测试: {test_name}")
        print(f"   运行次数: {num_runs}")
        print(f"   测试数据: {input_file}")
        
        times = []
        successful_runs = 0
        
        for i in range(num_runs):
            start_time = time.time()
            try:
                result = subprocess.run([
                    str(self.executable),
                    "realistic_end_to_end_anomaly_detector.dlc",
                    "realistic_end_to_end_anomaly_classifier.dlc",
                    str(input_path)
                ], capture_output=True, text=True, timeout=10)
                
                end_time = time.time()
                inference_time_ms = (end_time - start_time) * 1000
                
                if result.returncode == 0:
                    times.append(inference_time_ms)
                    successful_runs += 1
                    if (i + 1) % 5 == 0:
                        print(f"   进度: {i+1}/{num_runs} ({inference_time_ms:.1f}ms)")
                else:
                    print(f"   运行 {i+1} 失败")
                    
            except subprocess.TimeoutExpired:
                print(f"   运行 {i+1} 超时")
            except Exception as e:
                print(f"   运行 {i+1} 异常: {str(e)[:30]}")
        
        if len(times) < num_runs * 0.8:  # 成功率低于80%
            print(f"⚠️  成功率过低: {len(times)}/{num_runs}")
            return None
            
        # 计算统计信息
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput_fps = 1000 / avg_time if avg_time > 0 else 0
        
        details = f"成功率: {successful_runs}/{num_runs}"
        
        self.log_performance(test_name, avg_time, min_time, max_time, 
                           std_dev, throughput_fps, details)
        
        return {
            "avg_time_ms": avg_time,
            "throughput_fps": throughput_fps,
            "success_rate": successful_runs / num_runs
        }
    
    def test_different_scenarios(self):
        """测试不同场景下的性能"""
        print("🎯 开始多场景性能测试")
        print("=" * 60)
        
        scenarios = [
            ("正常网络状况", "normal_input.bin"),
            ("WiFi信号衰减", "wifi_degradation_input.bin"),
            ("网络延迟异常", "network_latency_input.bin"),
            ("连接中断异常", "connection_interruption_input.bin"),
            ("DNS解析异常", "dns_resolution_input.bin"),
            ("带宽限制异常", "bandwidth_limitation_input.bin")
        ]
        
        results = []
        
        for scenario_name, input_file in scenarios:
            result = self.run_inference_batch(scenario_name, input_file, num_runs=10)
            if result:
                results.append({
                    "scenario": scenario_name,
                    "result": result
                })
        
        return results
    
    def test_cold_vs_warm_start(self):
        """测试冷启动 vs 热启动性能"""
        print("🔥 测试冷启动 vs 热启动性能")
        print("=" * 60)
        
        # 冷启动测试 - 杀死所有相关进程后重新开始
        print("❄️  冷启动测试...")
        subprocess.run(["pkill", "-f", "dlc_mobile_inference"], capture_output=True)
        time.sleep(2)  # 确保进程完全停止
        
        cold_start_result = self.run_inference_batch("冷启动", "normal_input.bin", num_runs=5)
        
        # 热启动测试 - 连续运行
        print("🔥 热启动测试...")
        warm_start_result = self.run_inference_batch("热启动", "normal_input.bin", num_runs=10)
        
        if cold_start_result and warm_start_result:
            improvement = ((cold_start_result["avg_time_ms"] - warm_start_result["avg_time_ms"]) 
                         / cold_start_result["avg_time_ms"] * 100)
            print(f"📈 热启动性能提升: {improvement:.1f}%")
            
        return cold_start_result, warm_start_result
    
    def test_memory_usage(self):
        """测试内存使用情况"""
        print("🧠 测试内存使用情况")
        print("=" * 60)
        
        try:
            # 启动推理进程并监控内存
            process = subprocess.Popen([
                str(self.executable),
                "realistic_end_to_end_anomaly_detector.dlc",
                "realistic_end_to_end_anomaly_classifier.dlc",
                str(self.project_root / "normal_input.bin")
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 获取进程内存信息
            pid = process.pid
            time.sleep(0.1)  # 等待进程启动
            
            try:
                memory_info = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "pid,vsz,rss,pmem,cmd"],
                    capture_output=True, text=True, timeout=5
                )
                
                if memory_info.returncode == 0:
                    lines = memory_info.stdout.strip().split('\n')
                    if len(lines) > 1:
                        fields = lines[1].split()
                        vsz_kb = int(fields[1])  # 虚拟内存 KB
                        rss_kb = int(fields[2])  # 物理内存 KB
                        cpu_percent = fields[3]   # CPU百分比
                        
                        print(f"💾 内存使用情况:")
                        print(f"   虚拟内存: {vsz_kb/1024:.1f} MB")
                        print(f"   物理内存: {rss_kb/1024:.1f} MB")
                        print(f"   CPU使用率: {cpu_percent}%")
                        
                        self.results["memory_usage"] = {
                            "virtual_memory_mb": vsz_kb / 1024,
                            "physical_memory_mb": rss_kb / 1024,
                            "cpu_percent": cpu_percent
                        }
                
            except Exception as e:
                print(f"⚠️  内存监控失败: {str(e)}")
                
            # 等待进程完成
            process.wait(timeout=10)
            
        except Exception as e:
            print(f"❌ 内存测试失败: {str(e)}")
    
    def generate_performance_report(self):
        """生成性能报告"""
        print("\n📊 生成性能报告")
        print("=" * 60)
        
        if not self.results["performance_tests"]:
            print("❌ 无性能测试数据")
            return
            
        # 计算总体统计
        all_avg_times = [test["avg_time_ms"] for test in self.results["performance_tests"]]
        all_throughputs = [test["throughput_fps"] for test in self.results["performance_tests"]]
        
        self.results["summary"] = {
            "total_tests": len(self.results["performance_tests"]),
            "overall_avg_latency_ms": statistics.mean(all_avg_times),
            "best_latency_ms": min(all_avg_times),
            "worst_latency_ms": max(all_avg_times),
            "overall_avg_throughput_fps": statistics.mean(all_throughputs),
            "best_throughput_fps": max(all_throughputs)
        }
        
        # 保存详细报告
        report_file = self.project_root / "snpe_performance_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 显示摘要
        summary = self.results["summary"]
        print(f"📋 性能摘要:")
        print(f"   测试场景数: {summary['total_tests']}")
        print(f"   平均延迟: {summary['overall_avg_latency_ms']:.1f}ms")
        print(f"   最佳延迟: {summary['best_latency_ms']:.1f}ms")
        print(f"   最差延迟: {summary['worst_latency_ms']:.1f}ms")
        print(f"   平均吞吐量: {summary['overall_avg_throughput_fps']:.1f} FPS")
        print(f"   最高吞吐量: {summary['best_throughput_fps']:.1f} FPS")
        
        print(f"\n📝 详细报告已保存到: {report_file}")
        
        # 性能评级
        avg_latency = summary['overall_avg_latency_ms']
        if avg_latency < 20:
            grade = "优秀 (< 20ms)"
        elif avg_latency < 50:
            grade = "良好 (< 50ms)"
        elif avg_latency < 100:
            grade = "一般 (< 100ms)"
        else:
            grade = "需要优化 (>= 100ms)"
            
        print(f"🏆 性能评级: {grade}")
        
    def run_full_performance_test(self):
        """运行完整性能测试"""
        print("🚀 SNPE性能测试开始")
        print("=" * 60)
        
        # 1. 多场景性能测试
        scenario_results = self.test_different_scenarios()
        
        # 2. 冷热启动对比
        cold_result, warm_result = self.test_cold_vs_warm_start()
        
        # 3. 内存使用测试
        self.test_memory_usage()
        
        # 4. 生成报告
        self.generate_performance_report()
        
        return len(scenario_results) > 0

def main():
    tester = SNPEPerformanceTester()
    success = tester.run_full_performance_test()
    
    if success:
        print("\n✅ SNPE性能测试完成")
    else:
        print("\n❌ SNPE性能测试失败")
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 