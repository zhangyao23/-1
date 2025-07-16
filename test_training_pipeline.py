#!/usr/bin/env python3
"""
测试训练和转换流程的脚本
"""

import os
import sys
import time

def test_environment():
    """测试环境依赖"""
    print("🔍 测试环境依赖...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn未安装")
        return False
    
    try:
        import onnx
        print(f"✅ ONNX: 已安装")
    except ImportError:
        print("❌ ONNX未安装")
        return False
    
    # 检查SNPE工具
    snpe_tool = "2.26.2.240911/bin/x86_64-linux-clang/snpe-onnx-to-dlc"
    if os.path.exists(snpe_tool):
        print(f"✅ SNPE工具: {snpe_tool}")
    else:
        print(f"❌ SNPE工具不存在: {snpe_tool}")
        return False
    
    return True

def test_model_import():
    """测试模型导入"""
    print("\n🔍 测试模型导入...")
    
    try:
        from train_multitask_model import MultiTaskAnomalyModel
        model = MultiTaskAnomalyModel()
        print("✅ 多任务模型类导入成功")
        
        # 测试前向传播
        import torch
        model.eval()  # 设置为评估模式
        dummy_input = torch.randn(1, 11)
        output = model(dummy_input)
        print(f"✅ 模型前向传播成功，输出形状: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        return False

def test_data_generation():
    """测试数据生成"""
    print("\n🔍 测试数据生成...")
    
    try:
        from train_realistic_end_to_end_networks import generate_realistic_network_data
        
        # 生成少量数据用于测试
        X, y_binary, y_multiclass = generate_realistic_network_data(n_samples=100)
        print(f"✅ 数据生成成功")
        print(f"   输入数据形状: {X.shape}")
        print(f"   二分类标签形状: {y_binary.shape}")
        print(f"   多分类标签形状: {y_multiclass.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 数据生成失败: {e}")
        return False

def test_conversion_script():
    """测试转换脚本"""
    print("\n🔍 测试转换脚本...")
    
    try:
        from convert_pytorch_to_dlc import load_multitask_model, convert_to_onnx, convert_to_dlc
        print("✅ 转换脚本导入成功")
        return True
    except Exception as e:
        print(f"❌ 转换脚本导入失败: {e}")
        return False

def quick_training_test():
    """快速训练测试"""
    print("\n🚀 开始快速训练测试...")
    
    try:
        from train_multitask_model import MultiTaskAnomalyModel, generate_and_prepare_data
        
        # 准备数据
        print("   准备数据...")
        train_loader, test_loader = generate_and_prepare_data()
        
        # 创建模型
        print("   创建模型...")
        model = MultiTaskAnomalyModel()
        
        # 快速训练（只训练几个epoch）
        print("   开始快速训练（5个epoch）...")
        from train_multitask_model import train_model
        train_model(model, train_loader, test_loader, epochs=5)
        
        # 保存模型
        print("   保存模型...")
        import torch
        torch.save(model.state_dict(), 'multitask_model.pth')
        
        print("✅ 快速训练测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 快速训练测试失败: {e}")
        return False

def test_conversion():
    """测试模型转换"""
    print("\n🔄 测试模型转换...")
    
    if not os.path.exists('multitask_model.pth'):
        print("❌ 训练好的模型文件不存在")
        return False
    
    try:
        from convert_pytorch_to_dlc import convert_to_onnx, convert_to_dlc, load_multitask_model
        import torch
        # 加载模型结构和权重
        model = load_multitask_model()
        if model is None:
            print("❌ 无法加载训练好的模型")
            return False
        
        # 转换为ONNX
        print("   转换为ONNX...")
        if convert_to_onnx(model, "test_model.onnx"):
            print("✅ ONNX转换成功")
        else:
            print("❌ ONNX转换失败")
            return False
        
        # 转换为DLC
        print("   转换为DLC...")
        if convert_to_dlc("test_model.onnx", "test_model.dlc"):
            print("✅ DLC转换成功")
        else:
            print("❌ DLC转换失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 模型转换失败: {e}")
        return False

def cleanup_test_files():
    """清理测试文件"""
    test_files = ['test_model.onnx', 'test_model.dlc']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   清理测试文件: {file}")

def main():
    """主测试函数"""
    print("🧪 AI网络异常检测系统 - 训练转换流程测试")
    print("=" * 60)
    
    # 测试环境
    if not test_environment():
        print("\n❌ 环境测试失败，请检查依赖安装")
        return False
    
    # 测试模型导入
    if not test_model_import():
        print("\n❌ 模型导入测试失败")
        return False
    
    # 测试数据生成
    if not test_data_generation():
        print("\n❌ 数据生成测试失败")
        return False
    
    # 测试转换脚本
    if not test_conversion_script():
        print("\n❌ 转换脚本测试失败")
        return False
    
    # 快速训练测试
    if not quick_training_test():
        print("\n❌ 快速训练测试失败")
        return False
    
    # 测试转换
    if not test_conversion():
        print("\n❌ 模型转换测试失败")
        return False
    
    # 清理测试文件
    cleanup_test_files()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！训练和转换流程可以正常运行")
    print("\n📋 完整流程:")
    print("1. python3 train_multitask_model.py")
    print("2. python3 convert_pytorch_to_dlc.py")
    print("3. 使用生成的 multitask_model.dlc 进行部署")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 