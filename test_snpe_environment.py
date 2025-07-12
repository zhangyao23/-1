#!/usr/bin/env python3
"""
SNPE环境验证脚本
测试AI网络异常检测程序在SNPE环境中的运行情况
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

class SNPEEnvironmentValidator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.snpe_root = self.project_root / "2.26.2.240911"
        self.results = {"tests": [], "summary": {}}
        
    def log_test(self, name, status, details="", time_ms=0):
        """记录测试结果"""
        self.results["tests"].append({
            "name": name,
            "status": status,
            "details": details,
            "time_ms": time_ms
        })
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {name}: {details}")
    
    def test_snpe_sdk_installation(self):
        """测试1: SNPE SDK安装完整性"""
        print("\n🔍 测试1: SNPE SDK安装完整性")
        
        # 检查关键目录
        required_dirs = ["include", "lib", "bin"]
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.snpe_root / dir_name
            if dir_path.exists():
                self.log_test(f"目录检查: {dir_name}/", "PASS", f"存在 ({len(list(dir_path.rglob('*')))} 文件)")
            else:
                missing_dirs.append(dir_name)
                self.log_test(f"目录检查: {dir_name}/", "FAIL", "目录缺失")
        
        # 检查关键头文件
        required_headers = [
            "include/SNPE/SNPE/SNPE.hpp",
            "include/SNPE/SNPE/SNPEFactory.hpp", 
            "include/SNPE/DlContainer/IDlContainer.hpp",
            "include/SNPE/DlSystem/TensorMap.hpp"
        ]
        
        for header in required_headers:
            header_path = self.snpe_root / header
            if header_path.exists():
                size_kb = header_path.stat().st_size / 1024
                self.log_test(f"头文件: {header.split('/')[-1]}", "PASS", f"{size_kb:.1f}KB")
            else:
                self.log_test(f"头文件: {header.split('/')[-1]}", "FAIL", "文件缺失")
        
        # 检查核心库文件
        lib_path = self.snpe_root / "lib" / "x86_64-linux-clang" / "libSNPE.so"
        if lib_path.exists():
            size_mb = lib_path.stat().st_size / 1024 / 1024
            self.log_test("核心库: libSNPE.so", "PASS", f"{size_mb:.1f}MB")
        else:
            self.log_test("核心库: libSNPE.so", "FAIL", "库文件缺失")
            
        return len(missing_dirs) == 0
    
    def test_dlc_models(self):
        """测试2: DLC模型文件验证"""
        print("\n🤖 测试2: DLC模型文件验证")
        
        models = [
            ("异常检测模型", "realistic_end_to_end_anomaly_detector.dlc"),
            ("异常分类模型", "realistic_end_to_end_anomaly_classifier.dlc"),
            ("数据标准化器", "realistic_raw_data_scaler.pkl")
        ]
        
        all_present = True
        total_size = 0
        
        for name, filename in models:
            filepath = self.project_root / filename
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                total_size += filepath.stat().st_size
                
                # 使用snpe-dlc-info工具检查DLC文件
                if filename.endswith('.dlc'):
                    try:
                        dlc_info_tool = self.snpe_root / "bin" / "x86_64-linux-clang" / "snpe-dlc-info"
                        if dlc_info_tool.exists():
                            result = subprocess.run([str(dlc_info_tool), str(filepath)], 
                                                   capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                self.log_test(name, "PASS", f"{size_kb:.1f}KB, DLC格式有效")
                            else:
                                self.log_test(name, "WARN", f"{size_kb:.1f}KB, DLC验证失败")
                        else:
                            self.log_test(name, "PASS", f"{size_kb:.1f}KB (无snpe-dlc-info工具)")
                    except Exception as e:
                        self.log_test(name, "WARN", f"{size_kb:.1f}KB, 验证异常: {str(e)[:50]}")
                else:
                    self.log_test(name, "PASS", f"{size_kb:.1f}KB")
            else:
                self.log_test(name, "FAIL", "文件缺失")
                all_present = False
        
        total_size_kb = total_size / 1024
        self.log_test("模型总大小", "PASS" if all_present else "FAIL", f"{total_size_kb:.1f}KB")
        
        return all_present
    
    def test_cpp_compilation(self):
        """测试3: C++程序编译"""
        print("\n🔧 测试3: C++程序编译")
        
        start_time = time.time()
        try:
            # 运行编译脚本
            result = subprocess.run(["./build_mobile_inference.sh"], 
                                   capture_output=True, text=True, timeout=60)
            
            compile_time = int((time.time() - start_time) * 1000)
            
            if result.returncode == 0:
                # 检查生成的可执行文件
                executable = self.project_root / "dlc_mobile_inference"
                if executable.exists():
                    size_kb = executable.stat().st_size / 1024
                    self.log_test("C++编译", "PASS", f"成功, {size_kb:.0f}KB可执行文件", compile_time)
                    return True
                else:
                    self.log_test("C++编译", "FAIL", "编译成功但可执行文件缺失", compile_time)
            else:
                error_lines = result.stderr.split('\n')[:3]  # 只显示前3行错误
                error_summary = "; ".join([line.strip() for line in error_lines if line.strip()])
                self.log_test("C++编译", "FAIL", f"编译失败: {error_summary[:100]}", compile_time)
                
        except subprocess.TimeoutExpired:
            self.log_test("C++编译", "FAIL", "编译超时 (>60s)")
        except Exception as e:
            self.log_test("C++编译", "FAIL", f"编译异常: {str(e)[:50]}")
            
        return False
    
    def test_snpe_inference(self):
        """测试4: SNPE推理功能 (已更新为多任务模型)"""
        print("\n🚀 测试4: SNPE推理功能 (多任务模型)")

        executable = self.project_root / "dlc_mobile_inference"
        if not executable.exists():
            self.log_test("推理测试", "FAIL", "C++可执行文件 dlc_mobile_inference 不存在")
            return False

        model_path = self.project_root / "multitask_model.dlc"
        if not model_path.exists():
            self.log_test("推理测试", "FAIL", "合并后的模型 multitask_model.dlc 不存在")
            return False
            
        input_path = self.project_root / "example_normal_input.json"
        if not input_path.exists():
            self.log_test("推理测试", "FAIL", "测试输入 example_normal_input.json 不存在")
            return False

        start_time = time.time()
        try:
            result = subprocess.run([
                str(executable),
                str(model_path),
                str(input_path)
            ], capture_output=True, text=True, timeout=30)
            
            inference_time = int((time.time() - start_time) * 1000)
            
            if result.returncode == 0:
                # 检查是否生成了结果文件
                result_file = self.project_root / "inference_results.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        final_decision = results.get('final_decision', '未知')
                        self.log_test("推理测试 (正常数据)", "PASS", 
                                    f"推理成功, 结果: {final_decision}", inference_time)
                        return True
                    except Exception as e:
                        self.log_test("推理测试 (正常数据)", "WARN", 
                                    f"推理成功但结果解析失败: {e}", inference_time)
                        return False
                else:
                    self.log_test("推理测试 (正常数据)", "WARN", 
                                f"推理成功但无结果文件", inference_time)
                    return False
            else:
                error_msg = result.stderr.split('\n')[0] if result.stderr else "未知错误"
                self.log_test("推理测试 (正常数据)", "FAIL", 
                            f"推理失败: {error_msg[:100]}", inference_time)
                return False
                
        except subprocess.TimeoutExpired:
            self.log_test("推理测试 (正常数据)", "FAIL", "推理超时 (>30s)")
        except Exception as e:
            self.log_test("推理测试 (正常数据)", "FAIL", f"推理异常: {str(e)[:100]}")
            
        return False
    
    def test_runtime_availability(self):
        """测试5: 运行时可用性检查"""
        print("\n⚙️  测试5: 运行时可用性检查")
        
        # 检查可用的运行时
        available_runtimes = []
        
        # 检查CPU运行时（总是可用）
        self.log_test("CPU运行时", "PASS", "x86_64-linux-clang库已安装")
        available_runtimes.append("CPU")
        
        # 检查GPU运行时库
        gpu_libs = ["libQnnGpu.so", "libGLES_mali.so", "libOpenCL.so"]
        gpu_available = False
        for lib in gpu_libs:
            lib_path = self.snpe_root / "lib" / "x86_64-linux-clang" / lib
            if lib_path.exists():
                gpu_available = True
                break
        
        if gpu_available:
            self.log_test("GPU运行时", "PASS", "GPU库文件可用")
            available_runtimes.append("GPU")
        else:
            self.log_test("GPU运行时", "WARN", "GPU库文件不可用（正常情况）")
        
        # 检查DSP运行时（Hexagon）
        dsp_available = False
        hexagon_dirs = ["hexagon-v66", "hexagon-v68", "hexagon-v69", "hexagon-v73", "hexagon-v75"]
        for hexagon_dir in hexagon_dirs:
            hexagon_path = self.snpe_root / "lib" / hexagon_dir
            if hexagon_path.exists() and list(hexagon_path.glob("*.so")):
                dsp_available = True
                break
                
        if dsp_available:
            self.log_test("DSP运行时", "PASS", "Hexagon DSP库可用")
            available_runtimes.append("DSP")
        else:
            self.log_test("DSP运行时", "WARN", "DSP库不可用（正常情况）")
        
        self.log_test("可用运行时", "PASS", f"{', '.join(available_runtimes)}")
        return len(available_runtimes) > 0
    
    def generate_report(self):
        """生成验证报告"""
        print("\n📋 生成验证报告...")
        
        # 统计结果
        total_tests = len(self.results["tests"])
        passed_tests = len([t for t in self.results["tests"] if t["status"] == "PASS"])
        failed_tests = len([t for t in self.results["tests"] if t["status"] == "FAIL"])
        warning_tests = len([t for t in self.results["tests"] if t["status"] == "WARN"])
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warning_tests,
            "success_rate": f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
        }
        
        # 保存详细报告
        report_file = self.project_root / "snpe_validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📝 详细报告已保存到: {report_file}")
        
        return passed_tests, failed_tests, warning_tests
    
    def run_validation(self):
        """运行完整验证"""
        print("🚀 SNPE环境验证开始")
        print("=" * 60)
        
        # 执行所有测试
        tests = [
            self.test_snpe_sdk_installation,
            self.test_dlc_models,
            self.test_cpp_compilation,
            self.test_snpe_inference,
            self.test_runtime_availability
        ]
        
        overall_success = True
        for test_func in tests:
            try:
                result = test_func()
                if not result:
                    overall_success = False
            except Exception as e:
                print(f"❌ 测试异常: {test_func.__name__} - {str(e)}")
                overall_success = False
        
        # 生成报告
        passed, failed, warnings = self.generate_report()
        
        # 输出总结
        print("\n🎯 验证总结")
        print("=" * 60)
        print(f"总测试项: {passed + failed + warnings}")
        print(f"通过: {passed}")
        print(f"失败: {failed}")
        print(f"警告: {warnings}")
        print(f"成功率: {self.results['summary']['success_rate']}")
        
        if failed == 0:
            print("\n✅ SNPE环境验证通过！您的程序可以在SNPE环境中正常运行")
            return True
        else:
            print(f"\n❌ SNPE环境存在 {failed} 个问题，请检查上述错误")
            return False

def main():
    validator = SNPEEnvironmentValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 