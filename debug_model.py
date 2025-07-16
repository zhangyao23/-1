#!/usr/bin/env python3
"""
调试模型，检查权重分布和输出
"""
import torch
import numpy as np
from train_multitask_model import MultiTaskAnomalyModel

def debug_model():
    print("🔍 调试模型权重和输出")
    print("=" * 50)
    
    # 加载模型
    model = MultiTaskAnomalyModel()
    model.load_state_dict(torch.load("multitask_model.pth", map_location='cpu'))
    model.eval()
    
    # 检查分类头权重
    print("📊 分类头权重统计:")
    classification_weights = model.classification_head.weight.data
    print(f"  权重形状: {classification_weights.shape}")
    print(f"  权重均值: {classification_weights.mean():.6f}")
    print(f"  权重标准差: {classification_weights.std():.6f}")
    print(f"  权重范围: [{classification_weights.min():.6f}, {classification_weights.max():.6f}]")
    
    # 检查分类头偏置
    classification_bias = model.classification_head.bias.data
    print(f"  偏置均值: {classification_bias.mean():.6f}")
    print(f"  偏置标准差: {classification_bias.std():.6f}")
    print(f"  偏置范围: [{classification_bias.min():.6f}, {classification_bias.max():.6f}]")
    
    # 测试不同输入
    print("\n🧪 测试不同输入:")
    
    # 创建6种典型的异常输入
    test_inputs = [
        # wifi_degradation
        torch.tensor([20.0, -80.0, -60.0, 8000, 6000, 1500000, 1200000, 40.0, 50.0, 45.0, 30.0]),
        # network_latency  
        torch.tensor([70.0, -55.0, -85.0, 12000, 10000, 2500000, 2000000, 150.0, 180.0, 40.0, 25.0]),
        # connection_instability
        torch.tensor([40.0, -75.0, -65.0, 2000, 1500, 300000, 250000, 80.0, 100.0, 35.0, 20.0]),
        # bandwidth_congestion
        torch.tensor([85.0, -40.0, -95.0, 35000, 30000, 12000000, 10000000, 70.0, 60.0, 75.0, 60.0]),
        # system_stress
        torch.tensor([75.0, -50.0, -90.0, 14000, 11000, 2800000, 2300000, 30.0, 40.0, 95.0, 90.0]),
        # dns_issues
        torch.tensor([75.0, -50.0, -90.0, 15000, 12000, 3000000, 2500000, 25.0, 400.0, 40.0, 25.0])
    ]
    
    anomaly_types = [
        "wifi_degradation",
        "network_latency", 
        "connection_instability",
        "bandwidth_congestion",
        "system_stress",
        "dns_issues"
    ]
    
    for i, (input_tensor, expected_type) in enumerate(zip(test_inputs, anomaly_types)):
        print(f"\n【{expected_type}】:")
        
        # 获取原始输出
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
            classification_output = output[0, 2:8]
            
        print(f"  原始输出: {classification_output.numpy()}")
        
        # 计算softmax
        classification_probs = torch.softmax(classification_output, dim=0)
        predicted_class = torch.argmax(classification_probs).item()
        
        print(f"  Softmax概率: {classification_probs.numpy()}")
        print(f"  预测类别: {anomaly_types[predicted_class]} (索引: {predicted_class})")
        print(f"  期望类别: {expected_type} (索引: {i})")
        print(f"  是否正确: {'✅' if predicted_class == i else '❌'}")

if __name__ == "__main__":
    debug_model() 