#!/usr/bin/env python3
"""
将分别训练的异常检测和分类模型转换为DLC格式
"""
import torch
import torch.onnx
import numpy as np
from train_separate_models import AnomalyDetector, AnomalyClassifier

def convert_models_to_onnx():
    print("🔄 转换分别训练的模型为ONNX格式...")
    
    # 转换异常检测模型
    print("\n📊 转换异常检测模型...")
    detector = AnomalyDetector()
    detector.load_state_dict(torch.load("anomaly_detector.pth", map_location='cpu'))
    detector.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 11)
    
    # 导出为ONNX
    torch.onnx.export(
        detector,
        dummy_input,
        "anomaly_detector.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['detection_output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'detection_output': {0: 'batch_size'}
        }
    )
    print("✅ 异常检测模型已保存为 anomaly_detector.onnx")
    
    # 转换异常分类模型
    print("\n📊 转换异常分类模型...")
    classifier = AnomalyClassifier()
    classifier.load_state_dict(torch.load("anomaly_classifier.pth", map_location='cpu'))
    classifier.eval()
    
    # 导出为ONNX
    torch.onnx.export(
        classifier,
        dummy_input,
        "anomaly_classifier.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['classification_output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'classification_output': {0: 'batch_size'}
        }
    )
    print("✅ 异常分类模型已保存为 anomaly_classifier.onnx")

def create_dlc_conversion_script():
    print("\n📝 创建DLC转换脚本...")
    
    script_content = '''#!/bin/bash
# DLC转换脚本
# 需要安装SNPE SDK并设置环境变量

echo "🔄 转换异常检测模型为DLC格式..."

# 转换异常检测模型
snpe-onnx-to-dlc \
    -i anomaly_detector.onnx \
    -o anomaly_detector.dlc

echo "🔄 转换异常分类模型为DLC格式..."

# 转换异常分类模型  
snpe-onnx-to-dlc \
    -i anomaly_classifier.onnx \
    -o anomaly_classifier.dlc

echo "✅ DLC转换完成！"
echo "📁 生成的文件："
echo "   - anomaly_detector.dlc"
echo "   - anomaly_classifier.dlc"
'''
    
    with open('convert_to_dlc.sh', 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    import os
    os.chmod('convert_to_dlc.sh', 0o755)
    
    print("✅ DLC转换脚本已创建: convert_to_dlc.sh")
    print("💡 使用方法:")
    print("   1. 确保已安装SNPE SDK")
    print("   2. 运行: ./convert_to_dlc.sh")

def main():
    print("🚀 开始转换分别训练的模型...")
    
    # 转换为ONNX
    convert_models_to_onnx()
    
    # 创建DLC转换脚本
    create_dlc_conversion_script()
    
    print("\n🎉 转换完成！")
    print("📁 生成的文件：")
    print("   - anomaly_detector.onnx")
    print("   - anomaly_classifier.onnx") 
    print("   - convert_to_dlc.sh")
    print("\n💡 下一步：")
    print("   运行 ./convert_to_dlc.sh 转换为DLC格式")

if __name__ == "__main__":
    main() 