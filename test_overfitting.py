#!/usr/bin/env python3
"""
测试模型是否存在过拟合问题
"""
import torch
import numpy as np
import joblib
from train_separate_models import AnomalyDetector, AnomalyClassifier

def test_overfitting():
    print("🔍 测试模型过拟合情况")
    print("=" * 50)
    
    # 加载模型
    detector = AnomalyDetector()
    detector.load_state_dict(torch.load("anomaly_detector.pth", map_location='cpu'))
    detector.eval()
    
    classifier = AnomalyClassifier()
    classifier.load_state_dict(torch.load("anomaly_classifier.pth", map_location='cpu'))
    classifier.eval()
    
    # 加载标准化器
    scaler = joblib.load('separate_models_scaler.pkl')
    
    # 测试1: 边界模糊数据
    print("\n📊 测试1: 边界模糊数据")
    boundary_tests = [
        # 接近正常但可能异常的数据
        [65.0, -65.0, -85.0, 16000, 13000, 3500000, 2800000, 45.0, 60.0, 45.0, 30.0],
        [70.0, -60.0, -80.0, 17000, 14000, 3800000, 3200000, 50.0, 70.0, 50.0, 35.0],
        [75.0, -55.0, -75.0, 18000, 15000, 4200000, 3600000, 55.0, 80.0, 55.0, 40.0],
    ]
    
    for i, test_data in enumerate(boundary_tests):
        print(f"\n边界测试 {i+1}:")
        input_scaled = scaler.transform([test_data])
        input_tensor = torch.FloatTensor(input_scaled)
        
        with torch.no_grad():
            detection_output = detector(input_tensor)
            detection_probs = torch.softmax(detection_output, dim=1)
            is_anomaly = torch.argmax(detection_probs, dim=1).item()
            confidence = torch.max(detection_probs, dim=1)[0].item()
            
            print(f"  检测结果: {'异常' if is_anomaly == 1 else '正常'}")
            print(f"  置信度: {confidence:.4f}")
            print(f"  概率分布: {detection_probs[0].detach().numpy()}")
    
    # 测试2: 混合特征数据
    print("\n📊 测试2: 混合特征数据")
    mixed_tests = [
        # 部分特征异常，部分正常
        [85.0, -45.0, -90.0, 15000, 12000, 3000000, 2500000, 120.0, 250.0, 40.0, 25.0],  # 延迟异常
        [60.0, -75.0, -70.0, 15000, 12000, 3000000, 2500000, 25.0, 30.0, 85.0, 75.0],   # 信号+资源异常
        [75.0, -50.0, -90.0, 8000, 6000, 1500000, 1200000, 25.0, 30.0, 40.0, 25.0],     # 流量异常
    ]
    
    for i, test_data in enumerate(mixed_tests):
        print(f"\n混合测试 {i+1}:")
        input_scaled = scaler.transform([test_data])
        input_tensor = torch.FloatTensor(input_scaled)
        
        with torch.no_grad():
            detection_output = detector(input_tensor)
            detection_probs = torch.softmax(detection_output, dim=1)
            is_anomaly = torch.argmax(detection_probs, dim=1).item()
            confidence = torch.max(detection_probs, dim=1)[0].item()
            
            print(f"  检测结果: {'异常' if is_anomaly == 1 else '正常'}")
            print(f"  置信度: {confidence:.4f}")
            
            if is_anomaly == 1:
                classification_output = classifier(input_tensor)
                classification_probs = torch.softmax(classification_output, dim=1)
                predicted_class = torch.argmax(classification_probs, dim=1).item()
                class_confidence = torch.max(classification_probs, dim=1)[0].item()
                
                anomaly_types = ["wifi_degradation", "network_latency", "connection_instability", 
                               "bandwidth_congestion", "system_stress", "dns_issues"]
                predicted_type = anomaly_types[predicted_class] if predicted_class < len(anomaly_types) else f"未知({predicted_class})"
                
                print(f"  分类结果: {predicted_type}")
                print(f"  分类置信度: {class_confidence:.4f}")
    
    # 测试3: 噪声数据
    print("\n📊 测试3: 噪声数据")
    # 在正常数据基础上添加噪声
    base_normal = [85.0, -45.0, -90.0, 15000, 12000, 3000000, 2500000, 25.0, 30.0, 40.0, 25.0]
    
    for noise_level in [0.05, 0.1, 0.15, 0.2]:
        print(f"\n噪声水平 {noise_level*100}%:")
        noisy_data = []
        for val in base_normal:
            noise = np.random.normal(0, abs(val) * noise_level)
            noisy_data.append(val + noise)
        
        input_scaled = scaler.transform([noisy_data])
        input_tensor = torch.FloatTensor(input_scaled)
        
        with torch.no_grad():
            detection_output = detector(input_tensor)
            detection_probs = torch.softmax(detection_output, dim=1)
            is_anomaly = torch.argmax(detection_probs, dim=1).item()
            confidence = torch.max(detection_probs, dim=1)[0].item()
            
            print(f"  检测结果: {'异常' if is_anomaly == 1 else '正常'}")
            print(f"  置信度: {confidence:.4f}")
    
    print("\n🎯 过拟合分析总结:")
    print("如果模型在边界数据和混合特征上表现过于确定（置信度>0.9），")
    print("可能表明存在过拟合问题。")
    print("建议:")
    print("1. 增加数据复杂度")
    print("2. 添加更多正则化")
    print("3. 使用交叉验证")
    print("4. 测试真实网络数据")

if __name__ == "__main__":
    test_overfitting() 