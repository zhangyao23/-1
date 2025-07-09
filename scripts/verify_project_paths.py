#!/usr/bin/env python3
"""
项目路径验证脚本
检查所有必要的文件和目录是否存在，确保项目结构完整
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def get_project_root():
    """获取项目根目录"""
    current_dir = Path(__file__).resolve().parent
    # 向上查找项目根目录（包含README.md的目录）
    while current_dir != current_dir.parent:
        if (current_dir / "README.md").exists():
            return current_dir
        current_dir = current_dir.parent
    return Path.cwd()

def load_project_config(project_root):
    """加载项目配置"""
    config_file = project_root / "config" / "project_paths.conf"
    config = {}
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip().strip('"')
    
    return config

def check_critical_files(project_root, config):
    """检查关键文件是否存在"""
    print("🔍 检查关键文件...")
    
    critical_files = [
        ("C++推理程序", config.get("CPP_SOURCE_FILE", "dlc_mobile_inference.cpp")),
        ("编译脚本", config.get("BUILD_SCRIPT", "build_mobile_inference.sh")),
        ("异常检测DLC模型", config.get("DLC_DETECTOR_MODEL", "realistic_end_to_end_anomaly_detector.dlc")),
        ("异常分类DLC模型", config.get("DLC_CLASSIFIER_MODEL", "realistic_end_to_end_anomaly_classifier.dlc")),
        ("数据标准化器", config.get("DLC_SCALER_FILE", "realistic_raw_data_scaler.pkl")),
        ("README文档", config.get("README_FILE", "README.md")),
        ("测试数据生成器", config.get("TEST_INPUT_GENERATOR", "generate_test_input.py")),
        ("系统测试脚本", config.get("SYSTEM_TEST_SCRIPT", "test_realistic_end_to_end_system.py")),
    ]
    
    missing_files = []
    existing_files = []
    
    for name, filename in critical_files:
        filepath = project_root / filename
        if filepath.exists():
            size = filepath.stat().st_size
            size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
            existing_files.append((name, filename, size_str))
            print(f"✅ {name}: {filename} ({size_str})")
        else:
            missing_files.append((name, filename))
            print(f"❌ {name}: {filename} (缺失)")
    
    return existing_files, missing_files

def check_directories(project_root, config):
    """检查项目目录结构"""
    print("\n📁 检查目录结构...")
    
    required_dirs = [
        ("模型目录", config.get("MODELS_DIR", "models")),
        ("数据目录", config.get("DATA_DIR", "data")),
        ("测试目录", config.get("TEST_DIR", "test")),
        ("指南目录", config.get("GUIDE_DIR", "guide")),
        ("配置目录", config.get("CONFIG_DIR", "config")),
        ("源码目录", config.get("SRC_DIR", "src")),
    ]
    
    missing_dirs = []
    existing_dirs = []
    
    for name, dirname in required_dirs:
        dirpath = project_root / dirname
        if dirpath.exists() and dirpath.is_dir():
            file_count = len(list(dirpath.rglob("*")))
            existing_dirs.append((name, dirname, file_count))
            print(f"✅ {name}: {dirname}/ ({file_count} 文件)")
        else:
            missing_dirs.append((name, dirname))
            print(f"❌ {name}: {dirname}/ (缺失)")
    
    return existing_dirs, missing_dirs

def check_snpe_sdk(project_root, config):
    """检查SNPE SDK"""
    print("\n🔧 检查SNPE SDK...")
    
    snpe_path = project_root / config.get("SNPE_SDK_RELATIVE_PATH", "2.26.2.240911")
    
    if not snpe_path.exists():
        print(f"❌ SNPE SDK目录不存在: {snpe_path}")
        return False
    
    # 检查关键SNPE组件
    snpe_components = [
        ("include/SNPE", "头文件目录"),
        ("lib", "库文件目录"),
        ("include/SNPE/SNPE/SNPE.hpp", "核心头文件"),
        ("include/SNPE/SNPE/SNPEFactory.hpp", "工厂头文件"),
        ("include/SNPE/DlContainer/IDlContainer.hpp", "容器头文件"),
        ("include/SNPE/DlSystem/TensorMap.hpp", "张量映射头文件"),
    ]
    
    all_present = True
    for component, description in snpe_components:
        component_path = snpe_path / component
        if component_path.exists():
            print(f"✅ {description}: {component}")
        else:
            print(f"❌ {description}: {component} (缺失)")
            all_present = False
    
    # 检查库文件架构
    lib_path = snpe_path / "lib"
    if lib_path.exists():
        architectures = [d.name for d in lib_path.iterdir() if d.is_dir()]
        print(f"📋 可用架构: {', '.join(architectures)}")
    
    return all_present

def check_model_files(project_root, config):
    """检查模型文件完整性"""
    print("\n🤖 检查模型文件...")
    
    model_files = [
        config.get("DLC_DETECTOR_MODEL", "realistic_end_to_end_anomaly_detector.dlc"),
        config.get("DLC_CLASSIFIER_MODEL", "realistic_end_to_end_anomaly_classifier.dlc"),
        config.get("DLC_SCALER_FILE", "realistic_raw_data_scaler.pkl"),
    ]
    
    total_size = 0
    all_present = True
    
    for model_file in model_files:
        filepath = project_root / model_file
        if filepath.exists():
            size = filepath.stat().st_size
            size_kb = size / 1024
            total_size += size
            print(f"✅ {model_file}: {size_kb:.1f}KB")
        else:
            print(f"❌ {model_file}: 缺失")
            all_present = False
    
    print(f"📊 模型文件总大小: {total_size/1024:.1f}KB")
    expected_size = int(config.get("EXPECTED_MODEL_SIZE_KB", "248"))
    
    if abs(total_size/1024 - expected_size) < 50:  # 50KB容差
        print(f"✅ 模型大小符合预期 (~{expected_size}KB)")
    else:
        print(f"⚠️  模型大小与预期不符 (预期: {expected_size}KB, 实际: {total_size/1024:.1f}KB)")
    
    return all_present

def check_compilation_environment():
    """检查编译环境"""
    print("\n⚙️  检查编译环境...")
    
    commands = [
        ("g++", "GNU C++ 编译器"),
        ("python3", "Python 3"),
        ("make", "Make 构建工具"),
    ]
    
    available_tools = []
    missing_tools = []
    
    for cmd, description in commands:
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                available_tools.append((cmd, description, version_line))
                print(f"✅ {description}: {version_line}")
            else:
                missing_tools.append((cmd, description))
                print(f"❌ {description}: 未找到")
        except FileNotFoundError:
            missing_tools.append((cmd, description))
            print(f"❌ {description}: 未安装")
    
    return available_tools, missing_tools

def generate_report(project_root, config, checks):
    """生成验证报告"""
    print("\n📋 生成验证报告...")
    
    report = {
        "project_info": {
            "name": config.get("PROJECT_NAME", "AI Network Anomaly Detection System"),
            "version": config.get("PROJECT_VERSION", "2.0"),
            "root_path": str(project_root),
            "check_time": subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
        },
        "checks": checks,
        "summary": {
            "total_checks": len(checks),
            "passed_checks": sum(1 for check in checks if check.get("status") == "passed"),
            "failed_checks": sum(1 for check in checks if check.get("status") == "failed"),
            "warnings": sum(1 for check in checks if check.get("status") == "warning")
        }
    }
    
    report_file = project_root / "verification_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📝 验证报告已保存到: {report_file}")
    return report

def main():
    """主函数"""
    print("🚀 项目路径验证工具")
    print("=" * 50)
    
    # 获取项目根目录
    project_root = get_project_root()
    print(f"📍 项目根目录: {project_root}")
    
    # 加载配置
    config = load_project_config(project_root)
    print(f"📝 配置文件: {'已加载' if config else '使用默认配置'}")
    
    # 执行各项检查
    checks = []
    
    # 检查关键文件
    existing_files, missing_files = check_critical_files(project_root, config)
    checks.append({
        "name": "关键文件检查",
        "status": "passed" if not missing_files else "failed",
        "details": {"existing": len(existing_files), "missing": len(missing_files)}
    })
    
    # 检查目录结构
    existing_dirs, missing_dirs = check_directories(project_root, config)
    checks.append({
        "name": "目录结构检查",
        "status": "passed" if not missing_dirs else "failed",
        "details": {"existing": len(existing_dirs), "missing": len(missing_dirs)}
    })
    
    # 检查SNPE SDK
    snpe_ok = check_snpe_sdk(project_root, config)
    checks.append({
        "name": "SNPE SDK检查",
        "status": "passed" if snpe_ok else "failed",
        "details": {"sdk_present": snpe_ok}
    })
    
    # 检查模型文件
    models_ok = check_model_files(project_root, config)
    checks.append({
        "name": "模型文件检查",
        "status": "passed" if models_ok else "failed",
        "details": {"models_present": models_ok}
    })
    
    # 检查编译环境
    available_tools, missing_tools = check_compilation_environment()
    checks.append({
        "name": "编译环境检查",
        "status": "passed" if not missing_tools else "warning",
        "details": {"available": len(available_tools), "missing": len(missing_tools)}
    })
    
    # 生成报告
    report = generate_report(project_root, config, checks)
    
    # 输出总结
    print("\n🎯 验证总结")
    print("=" * 50)
    print(f"总检查项: {report['summary']['total_checks']}")
    print(f"通过: {report['summary']['passed_checks']}")
    print(f"失败: {report['summary']['failed_checks']}")
    print(f"警告: {report['summary']['warnings']}")
    
    if report['summary']['failed_checks'] > 0:
        print("\n❌ 项目存在问题，请修复后重试")
        sys.exit(1)
    elif report['summary']['warnings'] > 0:
        print("\n⚠️  项目基本正常，但有些组件可能需要安装")
        sys.exit(0)
    else:
        print("\n✅ 项目验证通过，所有组件都已就绪")
        sys.exit(0)

if __name__ == "__main__":
    main() 