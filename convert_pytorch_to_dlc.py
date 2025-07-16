#!/usr/bin/env python3
"""
AI网络异常检测模型转换脚本
将训练好的PyTorch模型转换为目标板子可用的DLC格式
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

def load_multitask_model():
    """
    加载训练好的多任务模型
    """
    from train_multitask_model import MultiTaskAnomalyModel
    
    # 创建模型实例
    model = MultiTaskAnomalyModel()
    
    # 加载训练好的权重
    model_path = "multitask_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✅ 成功加载模型权重: {model_path}")
    else:
        print(f"⚠️  模型文件不存在: {model_path}")
        print("请先运行 train_multitask_model.py 训练模型")
        return None
    
    model.eval()
    return model

def convert_to_onnx(model, output_path="multitask_model.onnx"):
    """
    将PyTorch模型转换为ONNX格式
    """
    print(f"🔄 开始转换为ONNX格式...")
    
    # 创建示例输入
    dummy_input = torch.randn(1, 11)  # 11维输入
    
    # 导出为ONNX
    torch.onnx.export(
        model,
        (dummy_input,),  # 将tensor包装为tuple
        output_path,
        input_names=['input'],
        output_names=['combined_output'],  # 只导出一个输出名
        dynamic_axes={
            'input': {0: 'batch_size'},
            'combined_output': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX模型验证通过: {output_path}")
        print(f"📊 模型大小: {os.path.getsize(output_path) / 1024:.1f} KB")
        return True
    except Exception as e:
        print(f"❌ ONNX模型验证失败: {e}")
        return False

def convert_to_dlc(onnx_path, dlc_path="multitask_model.dlc", use_qairt=True):
    """
    将ONNX模型转换为DLC格式
    支持两种转换方式：
    - use_qairt=True: 使用qairt-converter (推荐，与幻灯片方法一致)
    - use_qairt=False: 使用snpe-onnx-to-dlc (传统方法)
    """
    print(f"🔄 开始转换为DLC格式...")
    
    # 检查SNPE环境
    snpe_root = "2.26.2.240911"
    if not os.path.exists(snpe_root):
        print(f"❌ SNPE SDK未找到: {snpe_root}")
        print("请确保SNPE SDK已正确安装")
        return False
    
    # 设置SNPE环境变量
    snpe_root_abs = os.path.abspath(snpe_root)
    os.environ['SNPE_ROOT'] = snpe_root_abs
    os.environ['PYTHONPATH'] = f"{snpe_root_abs}/lib/python:{os.environ.get('PYTHONPATH', '')}"
    
    # 选择转换工具
    if use_qairt:
        converter_name = "qairt-converter"
        converter_path = os.path.join(snpe_root, "bin", "x86_64-linux-clang", "qairt-converter")
        print(f"🔧 使用转换工具: {converter_name} (与幻灯片方法一致)")
    else:
        converter_name = "snpe-onnx-to-dlc"
        converter_path = os.path.join(snpe_root, "bin", "x86_64-linux-clang", "snpe-onnx-to-dlc")
        print(f"🔧 使用转换工具: {converter_name} (传统方法)")
    
    if not os.path.exists(converter_path):
        print(f"❌ 转换工具未找到: {converter_path}")
        return False
    
    # 执行转换
    import subprocess
    
    if use_qairt:
        # 使用qairt-converter (幻灯片中的方法)
        cmd = [
            converter_path,
            "-i", onnx_path,
            "--output_path", dlc_path,
            "--model_version", "1.0"
        ]
    else:
        # 使用snpe-onnx-to-dlc (传统方法)
        cmd = [
            converter_path,
            "-i", onnx_path,
            "-o", dlc_path,
            "--input_encoding", "input", "other"
        ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ DLC转换成功: {dlc_path}")
        print(f"📊 模型大小: {os.path.getsize(dlc_path) / 1024:.1f} KB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ DLC转换失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def validate_dlc_model(dlc_path="multitask_model.dlc"):
    """
    验证DLC模型文件
    """
    print(f"🔍 验证DLC模型文件...")
    
    if not os.path.exists(dlc_path):
        print(f"❌ DLC模型文件不存在: {dlc_path}")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(dlc_path)
    print(f"📊 文件大小: {file_size / 1024:.1f} KB")
    
    # 检查文件头（DLC文件是ZIP格式，以PK开头）
    with open(dlc_path, 'rb') as f:
        header = f.read(4)
        if header.startswith(b'PK'):
            print("✅ DLC文件格式验证通过（ZIP格式）")
            return True
        else:
            print("❌ DLC文件格式验证失败")
            return False

def main():
    """
    主转换流程
    """
    print("🚀 AI网络异常检测模型转换开始")
    print("=" * 50)
    
    # 1. 加载PyTorch模型
    model = load_multitask_model()
    if model is None:
        return False
    
    # 2. 转换为ONNX
    onnx_success = convert_to_onnx(model)
    if not onnx_success:
        return False
    
    # 3. 转换为DLC
    dlc_success = convert_to_dlc("multitask_model.onnx")
    if not dlc_success:
        return False
    
    # 4. 验证DLC模型
    validation_success = validate_dlc_model()
    
    print("=" * 50)
    if validation_success:
        print("🎉 模型转换完成！")
        print("📁 生成的文件:")
        print("   - multitask_model.onnx (ONNX格式)")
        print("   - multitask_model.dlc (DLC格式)")
        print("\n💡 下一步:")
        print("   将 multitask_model.dlc 文件复制到目标板子")
        print("   参考 guide/模型集成指南.md 了解集成方法")
        return True
    else:
        print("❌ 模型转换失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 