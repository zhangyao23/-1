#!/usr/bin/env python3
"""
测试分别训练的异常检测和分类模型
"""
import torch
import numpy as np
import joblib
from train_separate_models import AnomalyDetector, AnomalyClassifier

def test_separate_models():
    print("🧪 测试分别训练的模型")
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
    
    # 测试输入
    test_inputs = [
        # 正常情况
        [85.0, -45.0, -90.0, 15000, 12000, 3000000, 2500000, 25.0, 30.0, 40.0, 25.0],
        # 模糊情况 - 边界数据
        [58.0, -68.0, -75.0, 18000, 15000, 4000000, 3500000, 95.0, 180.0, 78.0, 85.0],
        # wifi_degradation
        [20.0, -80.0, -60.0, 8000, 6000, 1500000, 1200000, 40.0, 50.0, 45.0, 30.0],
        # network_latency  
        [70.0, -55.0, -85.0, 12000, 10000, 2500000, 2000000, 150.0, 180.0, 40.0, 25.0],
        # connection_instability
        [40.0, -75.0, -65.0, 2000, 1500, 300000, 250000, 80.0, 100.0, 35.0, 20.0],
        # bandwidth_congestion
        [85.0, -40.0, -95.0, 35000, 30000, 12000000, 10000000, 70.0, 60.0, 75.0, 60.0],
        # system_stress
        [75.0, -50.0, -90.0, 14000, 11000, 2800000, 2300000, 30.0, 40.0, 95.0, 90.0],
        # dns_issues
        [75.0, -50.0, -90.0, 15000, 12000, 3000000, 2500000, 25.0, 400.0, 40.0, 25.0]
    ]
    
    anomaly_types = [
        "normal",
        "ambiguous",
        "wifi_degradation",
        "network_latency", 
        "connection_instability",
        "bandwidth_congestion",
        "system_stress",
        "dns_issues"
    ]
    
    for i, (input_data, expected_type) in enumerate(zip(test_inputs, anomaly_types)):
        print(f"\n【{expected_type}】:")
        
        # 标准化输入
        input_scaled = scaler.transform([input_data])
        input_tensor = torch.FloatTensor(input_scaled)
        
        # 异常检测
        with torch.no_grad():
            detection_output = detector(input_tensor)
            detection_probs = torch.softmax(detection_output, dim=1)
            is_anomaly = torch.argmax(detection_probs, dim=1).item()
            
            print(f"  异常检测: {'异常' if is_anomaly == 1 else '正常'} (概率: {detection_probs[0].detach().numpy()})")
            
            # 如果是异常，进行分类
            if is_anomaly == 1:
                classification_output = classifier(input_tensor)
                classification_probs = torch.softmax(classification_output, dim=1)
                predicted_class = int(torch.argmax(classification_probs, dim=1).item())
                
                # 对于正常情况，期望检测为正常
                if expected_type == "normal":
                    print(f"  期望结果: 正常")
                    print(f"  是否正确: {'✅' if is_anomaly == 0 else '❌'}")
                elif expected_type == "ambiguous":
                    print(f"  模糊数据 - 检测结果: {'异常' if is_anomaly == 1 else '正常'}")
                    print(f"  注意: 这是边界数据，结果可能不确定")
                else:
                    # 异常分类器只处理6种异常类型（不包括normal和ambiguous）
                    anomaly_class_names = ["wifi_degradation", "network_latency", "connection_instability", 
                                         "bandwidth_congestion", "system_stress", "dns_issues"]
                    predicted_class_name = anomaly_class_names[predicted_class] if predicted_class < len(anomaly_class_names) else f"未知类型({predicted_class})"
                    print(f"  异常分类: {predicted_class_name} (索引: {predicted_class})")
                    print(f"  分类概率: {classification_probs[0].detach().numpy()}")
                    print(f"  期望类别: {expected_type} (索引: {i - 2})")
                    print(f"  是否正确: {'✅' if predicted_class == (i - 2) else '❌'}")
            else:
                if expected_type == "normal":
                    print("  期望结果: 正常")
                    print("  是否正确: ✅")
                elif expected_type == "ambiguous":
                    print("  模糊数据 - 检测结果: 正常")
                    print("  注意: 这是边界数据，结果可能不确定")
                else:
                    print("  跳过分类（检测为正常）")
                    print("  是否正确: ❌ (应该检测为异常)")

if __name__ == "__main__":
    test_separate_models() 