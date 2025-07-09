import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 超简化的端到端异常检测网络 (11维输入)
class UltraSimplifiedEndToEndAnomalyDetector(nn.Module):
    def __init__(self):
        super(UltraSimplifiedEndToEndAnomalyDetector, self).__init__()
        
        # 直接从11维输入到2分类输出，避免复杂的特征工程
        self.network = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# 超简化的端到端异常分类网络 (11维输入)
class UltraSimplifiedEndToEndAnomalyClassifier(nn.Module):
    def __init__(self, n_classes=6):
        super(UltraSimplifiedEndToEndAnomalyClassifier, self).__init__()
        
        # 直接从11维输入到6分类输出
        self.network = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

def generate_challenging_test_data(n_samples=1000):
    """生成更具挑战性的测试数据，包含边界情况和噪声"""
    print(f"生成 {n_samples} 个具有挑战性的测试样本...")
    
    np.random.seed(123)  # 使用不同的随机种子
    
    test_data = []
    test_labels_binary = []
    test_labels_multiclass = []
    
    for i in range(n_samples):
        sample_type = np.random.choice([
            'normal', 'borderline_normal', 'borderline_anomaly', 
            'noisy_normal', 'noisy_anomaly', 'mixed_signals'
        ])
        
        if sample_type == 'normal':
            # 标准正常样本
            sample = [
                np.random.uniform(70, 95),      # 信号质量
                np.random.uniform(-55, -30),    # 信号强度
                np.random.uniform(-95, -85),    # 噪声
                np.random.uniform(8000, 25000), # rx包
                np.random.uniform(6000, 20000), # tx包
                np.random.uniform(1000000, 8000000), # rx字节
                np.random.uniform(500000, 6000000),  # tx字节
                np.random.uniform(8, 25),       # ping
                np.random.uniform(15, 40),      # dns
                np.random.uniform(20, 60),      # 内存
                np.random.uniform(5, 30)        # CPU
            ]
            binary_label = 0
            multiclass_label = 0
            
        elif sample_type == 'borderline_normal':
            # 边界正常样本（接近异常阈值）
            sample = [
                np.random.uniform(65, 75),      # 信号质量稍差
                np.random.uniform(-60, -50),    # 信号强度稍弱
                np.random.uniform(-90, -80),    # 噪声稍高
                np.random.uniform(6000, 10000), # 包数偏少
                np.random.uniform(4000, 8000),  # 包数偏少
                np.random.uniform(800000, 1500000), # 字节数偏少
                np.random.uniform(400000, 1000000), # 字节数偏少
                np.random.uniform(20, 35),      # ping稍长
                np.random.uniform(30, 50),      # dns稍长
                np.random.uniform(50, 70),      # 内存稍高
                np.random.uniform(25, 40)       # CPU稍高
            ]
            binary_label = 0  # 仍然是正常
            multiclass_label = 0
            
        elif sample_type == 'borderline_anomaly':
            # 边界异常样本（轻微异常）
            sample = [
                np.random.uniform(50, 70),      # 信号质量差
                np.random.uniform(-70, -55),    # 信号强度差
                np.random.uniform(-80, -70),    # 噪声较高
                np.random.uniform(3000, 8000),  # 包数少
                np.random.uniform(2000, 6000),  # 包数少
                np.random.uniform(300000, 1000000), # 字节数少
                np.random.uniform(200000, 800000),  # 字节数少
                np.random.uniform(35, 60),      # ping较长
                np.random.uniform(45, 80),      # dns较长
                np.random.uniform(60, 80),      # 内存较高
                np.random.uniform(35, 55)       # CPU较高
            ]
            binary_label = 1  # 轻微异常
            multiclass_label = np.random.choice([1, 2, 3]) + 1  # 随机异常类型
            
        elif sample_type == 'noisy_normal':
            # 带噪声的正常样本
            base_sample = [85.0, -45.0, -90.0, 15000, 12000, 3000000, 2500000, 15.0, 25.0, 35.0, 20.0]
            sample = []
            for val in base_sample:
                # 添加10-20%的随机噪声
                noise = np.random.uniform(-0.2, 0.2) * abs(val)
                sample.append(val + noise)
            binary_label = 0
            multiclass_label = 0
            
        elif sample_type == 'noisy_anomaly':
            # 带噪声的异常样本
            base_sample = [10.0, -85.0, -45.0, 50, 30, 5000, 3000, 200.0, 300.0, 30.0, 15.0]
            sample = []
            for val in base_sample:
                # 添加15-25%的随机噪声
                noise = np.random.uniform(-0.25, 0.25) * abs(val) if val != 0 else np.random.uniform(-5, 5)
                sample.append(val + noise)
            binary_label = 1
            multiclass_label = 1  # wifi_disconnection
            
        elif sample_type == 'mixed_signals':
            # 混合信号样本（一些指标正常，一些异常）
            sample = [
                np.random.uniform(70, 90),      # 信号质量正常
                np.random.uniform(-50, -35),    # 信号强度正常
                np.random.uniform(-90, -85),    # 噪声正常
                np.random.uniform(100, 1000),   # 包数异常低
                np.random.uniform(80, 800),     # 包数异常低
                np.random.uniform(10000, 100000), # 字节数异常低
                np.random.uniform(8000, 80000),    # 字节数异常低
                np.random.uniform(10, 30),      # ping正常
                np.random.uniform(20, 50),      # dns正常
                np.random.uniform(25, 45),      # 内存正常
                np.random.uniform(10, 30)       # CPU正常
            ]
            binary_label = 1  # 应该被检测为异常
            multiclass_label = 2  # packet_loss
        
        test_data.append(sample)
        test_labels_binary.append(binary_label)
        test_labels_multiclass.append(multiclass_label)
    
    return np.array(test_data), np.array(test_labels_binary), np.array(test_labels_multiclass)

def test_model_robustness():
    """测试模型在具有挑战性数据上的表现"""
    print("=== 模型鲁棒性测试 ===")
    print()
    
    # 加载模型和标准化器
    scaler = joblib.load('ultra_simplified_raw_data_scaler.pkl')
    
    detector_model = UltraSimplifiedEndToEndAnomalyDetector()
    detector_model.load_state_dict(torch.load('ultra_simplified_end_to_end_anomaly_detector.pth', map_location='cpu'))
    detector_model.eval()
    
    classifier_model = UltraSimplifiedEndToEndAnomalyClassifier(n_classes=6)
    classifier_model.load_state_dict(torch.load('ultra_simplified_end_to_end_anomaly_classifier.pth', map_location='cpu'))
    classifier_model.eval()
    
    # 生成具有挑战性的测试数据
    test_data, test_labels_binary, test_labels_multiclass = generate_challenging_test_data(1000)
    
    # 标准化测试数据
    test_data_scaled = scaler.transform(test_data)
    test_tensor = torch.FloatTensor(test_data_scaled)
    
    # 测试异常检测
    print("🔍 **异常检测性能测试**")
    with torch.no_grad():
        detection_output = detector_model(test_tensor)
        detection_probs = torch.softmax(detection_output, dim=1)
        predicted_binary = torch.argmax(detection_output, dim=1).numpy()
    
    # 计算异常检测准确率
    detection_accuracy = accuracy_score(test_labels_binary, predicted_binary)
    print(f"   异常检测准确率: {detection_accuracy:.3f} ({detection_accuracy*100:.1f}%)")
    
    # 详细分析
    cm_binary = confusion_matrix(test_labels_binary, predicted_binary)
    print(f"   混淆矩阵:")
    print(f"   实际\\预测  正常   异常")
    print(f"   正常      {cm_binary[0,0]:4d}   {cm_binary[0,1]:4d}")
    print(f"   异常      {cm_binary[1,0]:4d}   {cm_binary[1,1]:4d}")
    
    # 置信度分布
    confidence_normal = detection_probs[test_labels_binary == 0, 0].numpy()
    confidence_anomaly = detection_probs[test_labels_binary == 1, 1].numpy()
    
    print(f"   正常样本置信度: 均值={confidence_normal.mean():.3f}, 最小={confidence_normal.min():.3f}")
    print(f"   异常样本置信度: 均值={confidence_anomaly.mean():.3f}, 最小={confidence_anomaly.min():.3f}")
    
    # 测试异常分类（仅对检测为异常的样本）
    anomaly_mask = predicted_binary == 1
    if np.sum(anomaly_mask) > 0:
        print()
        print("🎯 **异常分类性能测试**")
        
        with torch.no_grad():
            classification_output = classifier_model(test_tensor[anomaly_mask])
            predicted_multiclass = torch.argmax(classification_output, dim=1).numpy()
        
        # 转换真实标签（0=normal, 1-6=anomaly types -> 0-5=anomaly types）
        true_multiclass = test_labels_multiclass[anomaly_mask] - 1
        true_multiclass = np.clip(true_multiclass, 0, 5)  # 确保在0-5范围内
        
        classification_accuracy = accuracy_score(true_multiclass, predicted_multiclass)
        print(f"   异常分类准确率: {classification_accuracy:.3f} ({classification_accuracy*100:.1f}%)")
        
        # 分类报告
        print("   分类详细报告:")
        unique_classes = np.unique(np.concatenate([true_multiclass, predicted_multiclass]))
        class_names = ['wifi_disconnection', 'high_latency', 'packet_loss', 'bandwidth_saturation', 'system_overload', 'dns_failure']
        active_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
        
        if len(active_class_names) > 0:
            print(classification_report(true_multiclass, predicted_multiclass, 
                                      labels=unique_classes, target_names=active_class_names))
        else:
            print("   没有足够的分类数据进行分析")
    
    # 分析低置信度样本
    print()
    print("🔍 **低置信度样本分析**")
    low_confidence_threshold = 0.8
    
    # 异常检测低置信度
    detection_confidence = np.max(detection_probs.numpy(), axis=1)
    low_conf_mask = detection_confidence < low_confidence_threshold
    
    print(f"   置信度 < {low_confidence_threshold} 的样本: {np.sum(low_conf_mask)} / {len(test_data)} ({np.sum(low_conf_mask)/len(test_data)*100:.1f}%)")
    
    if np.sum(low_conf_mask) > 0:
        low_conf_accuracy = accuracy_score(test_labels_binary[low_conf_mask], predicted_binary[low_conf_mask])
        print(f"   低置信度样本准确率: {low_conf_accuracy:.3f} ({low_conf_accuracy*100:.1f}%)")

def main():
    print("=== 模型鲁棒性和挑战性测试 ===")
    print()
    
    print("🎯 **测试目标**:")
    print("   - 验证模型在边界情况下的表现")
    print("   - 测试对噪声数据的鲁棒性")
    print("   - 检查是否存在过拟合")
    print("   - 分析置信度分布")
    print()
    
    test_model_robustness()
    
    print()
    print("=== 测试分析结论 ===")
    print("如果准确率仍然接近100%，说明:")
    print("1. 可能存在过拟合")
    print("2. 测试数据可能仍然过于简单")
    print("3. 需要收集真实世界的数据进行验证")

if __name__ == "__main__":
    main() 