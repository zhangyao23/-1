#!/usr/bin/env python3
"""
使用训练数据中的异常样本测试模型多分类能力
"""
import torch
import numpy as np
from train_multitask_model import MultiTaskAnomalyModel
from train_realistic_end_to_end_networks import generate_realistic_network_data

def test_with_training_data():
    print("🔍 使用训练数据测试模型多分类能力")
    print("=" * 50)
    
    # 生成训练数据
    X, y_binary, y_multiclass = generate_realistic_network_data(n_samples=1000)
    
    # 加载模型
    model = MultiTaskAnomalyModel()
    model.load_state_dict(torch.load("multitask_model.pth", map_location='cpu'))
    model.eval()
    
    # 获取异常样本
    anomaly_indices = np.where(y_binary == 1)[0]
    anomaly_data = X[anomaly_indices]
    anomaly_labels = y_multiclass[anomaly_indices] - 1  # 转换为0-5
    
    anomaly_types = [
        "wifi_degradation",
        "network_latency", 
        "connection_instability",
        "bandwidth_congestion",
        "system_stress",
        "dns_issues"
    ]
    
    print(f"测试样本数: {len(anomaly_data)}")
    print(f"异常类型分布: {np.bincount(anomaly_labels)}")
    
    # 测试每个异常类型
    correct_predictions = 0
    total_predictions = 0
    
    for i, anomaly_type in enumerate(anomaly_types):
        type_indices = np.where(anomaly_labels == i)[0]
        if len(type_indices) == 0:
            continue
            
        type_data = anomaly_data[type_indices]
        type_labels = anomaly_labels[type_indices]
        
        print(f"\n【{anomaly_type}】样本数: {len(type_data)}")
        
        # 随机选择5个样本进行测试
        test_indices = np.random.choice(len(type_data), min(5, len(type_data)), replace=False)
        
        for idx in test_indices:
            sample = type_data[idx]
            true_label = type_labels[idx]
            
            # 推理
            input_tensor = torch.FloatTensor(sample).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
            
            detection_output = output[0, 0:2]
            classification_output = output[0, 2:8]
            
            detection_probs = torch.softmax(detection_output, dim=0)
            classification_probs = torch.softmax(classification_output, dim=0)
            
            predicted_class = torch.argmax(classification_probs).item()
            is_correct = predicted_class == true_label
            
            print(f"  真实: {anomaly_types[true_label]}, 预测: {anomaly_types[predicted_class]}, 正确: {'✅' if is_correct else '❌'}")
            print(f"  各类型概率: {[f'{anomaly_types[j]}={classification_probs[j]:.3f}' for j in range(6)]}")
            
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
    
    print(f"\n📊 测试结果汇总:")
    print(f"  总预测数: {total_predictions}")
    print(f"  正确预测: {correct_predictions}")
    print(f"  准确率: {correct_predictions/total_predictions*100:.2f}%")

if __name__ == "__main__":
    test_with_training_data() 