#!/usr/bin/env python3
"""
DLC输出处理工具
处理两阶段DLC系统的原始输出，生成标准化的JSON结果
"""

import numpy as np
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

class DLCOutputProcessor:
    """DLC输出处理器"""
    
    def __init__(self):
        self.anomaly_classes = {
            0: "wifi_degradation",      # WiFi信号衰减
            1: "network_latency",       # 网络延迟
            2: "connection_instability", # 连接不稳定
            3: "bandwidth_congestion",  # 带宽拥塞
            4: "system_stress",         # 系统压力
            5: "dns_issues"             # DNS问题
        }
        
        self.severity_mapping = {
            "wifi_degradation": "medium",
            "network_latency": "high",
            "connection_instability": "high",
            "bandwidth_congestion": "medium",
            "system_stress": "critical",
            "dns_issues": "medium"
        }
        
        self.action_recommendations = {
            "wifi_degradation": [
                "Check WiFi signal strength",
                "Move closer to router",
                "Check for interference sources"
            ],
            "network_latency": [
                "Check network connection",
                "Restart router", 
                "Contact ISP if problem persists",
                "Check for background downloads"
            ],
            "connection_instability": [
                "Check network cable connections",
                "Restart network adapter",
                "Update network drivers"
            ],
            "bandwidth_congestion": [
                "Close bandwidth-heavy applications",
                "Limit background updates",
                "Upgrade internet plan if needed"
            ],
            "system_stress": [
                "Close unnecessary applications",
                "Restart system",
                "Check for memory leaks",
                "Monitor resource usage"
            ],
            "dns_issues": [
                "Try different DNS servers",
                "Flush DNS cache",
                "Check DNS configuration"
            ]
        }
    
    def process_detection_output(self, raw_output: List[List[float]]) -> Dict[str, Any]:
        """
        处理异常检测DLC的原始输出
        
        Args:
            raw_output: DLC模型的原始输出 [[异常logit, 正常logit]]
            
        Returns:
            dict: 处理后的结果
        """
        # 验证输入格式
        if not isinstance(raw_output, list) or len(raw_output) != 1:
            raise ValueError("Detection output must be a list with one element")
        
        if not isinstance(raw_output[0], list) or len(raw_output[0]) != 2:
            raise ValueError("Detection output must have shape [1, 2]")
        
        logits = np.array(raw_output[0], dtype=np.float32)
        
        # 应用softmax获取概率
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
        probabilities = exp_logits / np.sum(exp_logits)
        
        # 获取预测结果
        predicted_class = int(np.argmax(probabilities))
        is_anomaly = predicted_class == 0  # 索引0代表异常
        confidence = float(np.max(probabilities))
        
        return {
            "raw_logits": logits.tolist(),
            "probabilities": probabilities.tolist(),
            "predicted_class": predicted_class,
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "anomaly_probability": float(probabilities[0]),
            "normal_probability": float(probabilities[1])
        }
    
    def process_classification_output(self, raw_output: List[List[float]]) -> Dict[str, Any]:
        """
        处理异常分类DLC的原始输出
        
        Args:
            raw_output: DLC模型的原始输出 [[6个异常类型的logit值]]
            
        Returns:
            dict: 处理后的结果
        """
        # 验证输入格式
        if not isinstance(raw_output, list) or len(raw_output) != 1:
            raise ValueError("Classification output must be a list with one element")
        
        if not isinstance(raw_output[0], list) or len(raw_output[0]) != 6:
            raise ValueError("Classification output must have shape [1, 6]")
        
        logits = np.array(raw_output[0], dtype=np.float32)
        
        # 应用softmax获取概率
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
        probabilities = exp_logits / np.sum(exp_logits)
        
        # 获取预测结果
        predicted_class_index = int(np.argmax(probabilities))
        predicted_class_name = self.anomaly_classes[predicted_class_index]
        confidence = float(np.max(probabilities))
        
        # 构建详细概率分布
        class_probabilities = {}
        for i, class_name in self.anomaly_classes.items():
            class_probabilities[class_name] = float(probabilities[i])
        
        return {
            "raw_logits": logits.tolist(),
            "probabilities": probabilities.tolist(),
            "predicted_class_index": predicted_class_index,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence,
            "class_probabilities": class_probabilities
        }
    
    def integrate_results(self, 
                         input_data: Optional[Dict] = None,
                         detection_result: Optional[Dict] = None, 
                         classification_result: Optional[Dict] = None,
                         processing_time_ms: Optional[float] = None) -> Dict[str, Any]:
        """
        整合两阶段系统的输出结果
        
        Args:
            input_data: 原始输入数据（可选）
            detection_result: 异常检测结果
            classification_result: 异常分类结果（可选）
            processing_time_ms: 处理时间（毫秒）
            
        Returns:
            dict: 最终的系统输出
        """
        integrated_result = {
            "timestamp": datetime.now().isoformat() + "Z",
            "processing_time_ms": processing_time_ms or 0.0
        }
        
        if detection_result:
            integrated_result["detection_stage"] = {
                "is_anomaly": detection_result["is_anomaly"],
                "confidence": detection_result["confidence"],
                "anomaly_probability": detection_result["anomaly_probability"],
                "normal_probability": detection_result["normal_probability"]
            }
            
            if detection_result["is_anomaly"] and classification_result:
                anomaly_type = classification_result["predicted_class_name"]
                integrated_result["classification_stage"] = {
                    "anomaly_type": anomaly_type,
                    "confidence": classification_result["confidence"],
                    "all_probabilities": classification_result["class_probabilities"]
                }
                
                # 添加最终结果和建议
                integrated_result["final_result"] = {
                    "status": "anomaly_detected",
                    "message": f"{anomaly_type.replace('_', ' ').title()} detected",
                    "severity": self.severity_mapping.get(anomaly_type, "medium"),
                    "action_required": True,
                    "recommended_actions": self.action_recommendations.get(anomaly_type, [])
                }
            else:
                integrated_result["classification_stage"] = None
                integrated_result["final_result"] = {
                    "status": "normal",
                    "message": "Network is operating normally",
                    "action_required": False
                }
        
        return integrated_result

def test_detection_examples():
    """测试异常检测输出处理示例"""
    processor = DLCOutputProcessor()
    
    print("🔍 **异常检测输出处理测试**")
    print("=" * 60)
    
    # 测试正常样本
    print("\n📱 **正常样本测试**:")
    normal_raw = [[-2.1543, 3.8967]]  # [异常logit, 正常logit]
    normal_result = processor.process_detection_output(normal_raw)
    
    print(f"   原始输出: {normal_raw[0]}")
    print(f"   是否异常: {normal_result['is_anomaly']}")
    print(f"   置信度: {normal_result['confidence']:.4f}")
    print(f"   异常概率: {normal_result['anomaly_probability']:.4f}")
    print(f"   正常概率: {normal_result['normal_probability']:.4f}")
    
    # 测试异常样本
    print("\n🚨 **异常样本测试**:")
    anomaly_raw = [[4.2156, -1.3547]]  # [异常logit, 正常logit]
    anomaly_result = processor.process_detection_output(anomaly_raw)
    
    print(f"   原始输出: {anomaly_raw[0]}")
    print(f"   是否异常: {anomaly_result['is_anomaly']}")
    print(f"   置信度: {anomaly_result['confidence']:.4f}")
    print(f"   异常概率: {anomaly_result['anomaly_probability']:.4f}")
    print(f"   正常概率: {anomaly_result['normal_probability']:.4f}")

def test_classification_examples():
    """测试异常分类输出处理示例"""
    processor = DLCOutputProcessor()
    
    print("\n🏷️ **异常分类输出处理测试**")
    print("=" * 60)
    
    # 测试WiFi信号衰减
    print("\n📶 **WiFi信号衰减测试**:")
    wifi_raw = [[3.2156, -1.1547, 0.8432, -0.5231, 1.2341, -2.1234]]
    wifi_result = processor.process_classification_output(wifi_raw)
    
    print(f"   原始输出: {[round(x, 3) for x in wifi_raw[0]]}")
    print(f"   预测类型: {wifi_result['predicted_class_name']}")
    print(f"   置信度: {wifi_result['confidence']:.4f}")
    print("   所有类型概率:")
    for class_name, prob in wifi_result['class_probabilities'].items():
        print(f"     {class_name}: {prob:.4f}")
    
    # 测试网络延迟
    print("\n🌐 **网络延迟测试**:")
    latency_raw = [[-0.8432, 4.1234, -1.2341, 0.2156, -0.5431, 1.3456]]
    latency_result = processor.process_classification_output(latency_raw)
    
    print(f"   原始输出: {[round(x, 3) for x in latency_raw[0]]}")
    print(f"   预测类型: {latency_result['predicted_class_name']}")
    print(f"   置信度: {latency_result['confidence']:.4f}")
    print("   前3类型概率:")
    sorted_probs = sorted(latency_result['class_probabilities'].items(), 
                         key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_probs[:3]:
        print(f"     {class_name}: {prob:.4f}")

def test_complete_pipeline():
    """测试完整的两阶段处理流程"""
    processor = DLCOutputProcessor()
    
    print("\n🔄 **完整两阶段流程测试**")
    print("=" * 60)
    
    # 模拟完整流程：WiFi异常
    print("\n📡 **WiFi异常完整流程**:")
    
    # 输入数据（示例）
    input_data = {
        "wlan0_wireless_quality": 45.0,
        "wlan0_signal_level": -70.0,
        "wlan0_noise_level": -75.0,
        "gateway_ping_time": 45.0,
        "dns_resolution_time": 60.0
    }
    
    # 阶段1：异常检测输出
    detection_raw = [[4.2156, -1.3547]]
    detection_result = processor.process_detection_output(detection_raw)
    
    print(f"   阶段1 - 异常检测:")
    print(f"     是否异常: {detection_result['is_anomaly']}")
    print(f"     置信度: {detection_result['confidence']:.4f}")
    
    # 阶段2：异常分类输出（仅在检测到异常时）
    classification_result = None
    if detection_result['is_anomaly']:
        classification_raw = [[3.2156, -1.1547, 0.8432, -0.5231, 1.2341, -2.1234]]
        classification_result = processor.process_classification_output(classification_raw)
        
        print(f"   阶段2 - 异常分类:")
        print(f"     异常类型: {classification_result['predicted_class_name']}")
        print(f"     置信度: {classification_result['confidence']:.4f}")
    
    # 整合最终结果
    final_result = processor.integrate_results(
        input_data=input_data,
        detection_result=detection_result,
        classification_result=classification_result,
        processing_time_ms=12.3
    )
    
    print(f"\n📋 **最终整合结果**:")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))

def test_error_handling():
    """测试错误处理"""
    processor = DLCOutputProcessor()
    
    print("\n⚠️ **错误处理测试**")
    print("=" * 60)
    
    # 测试格式错误
    error_cases = [
        ("检测输出维度错误", [[1.0, 2.0, 3.0]]),  # 应该是2维
        ("分类输出维度错误", [[1.0, 2.0, 3.0, 4.0, 5.0]]),  # 应该是6维
        ("输出格式错误", [1.0, 2.0]),  # 应该是嵌套列表
    ]
    
    for test_name, error_input in error_cases:
        print(f"\n❌ **{test_name}**:")
        try:
            if "检测" in test_name:
                processor.process_detection_output(error_input)
            else:
                processor.process_classification_output(error_input)
            print("   意外：没有抛出错误")
        except ValueError as e:
            print(f"   ✅ 正确捕获错误: {e}")
        except Exception as e:
            print(f"   ⚠️ 其他错误: {e}")

def main():
    """主函数"""
    print("🎯 **DLC输出处理工具**")
    print("处理两阶段DLC系统的原始输出，生成标准化JSON结果")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # 处理命令行提供的文件
        print("📁 **文件处理模式**")
        print("功能开发中...")
    else:
        # 运行内置示例测试
        test_detection_examples()
        test_classification_examples()
        test_complete_pipeline()
        test_error_handling()
        
        print("\n💡 **使用说明**:")
        print("   📖 查看详细格式规范: cat OUTPUT_FORMAT_SPECIFICATION.md")
        print("   🔧 处理自定义输出: 修改此脚本中的示例数据")
        print("   🎯 集成到应用: 使用DLCOutputProcessor类")
        
        print("\n📊 **输出格式要点**:")
        print("   ✅ 异常检测: [1, 2] 输出 → softmax → 异常/正常概率")
        print("   ✅ 异常分类: [1, 6] 输出 → softmax → 6种异常类型概率")
        print("   ✅ 最终结果: 结构化JSON + 行动建议")
        print("   ✅ 错误处理: 完整的输入验证和异常处理")

if __name__ == "__main__":
    main() 