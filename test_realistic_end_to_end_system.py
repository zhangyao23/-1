import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 真实端到端异常检测网络 (11维输入)
class RealisticEndToEndAnomalyDetector(nn.Module):
    def __init__(self):
        super(RealisticEndToEndAnomalyDetector, self).__init__()
        
        # 增加网络复杂度和正则化来处理真实数据
        self.network = nn.Sequential(
            nn.Linear(11, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
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

# 真实端到端异常分类网络 (11维输入)
class RealisticEndToEndAnomalyClassifier(nn.Module):
    def __init__(self, n_classes=6):
        super(RealisticEndToEndAnomalyClassifier, self).__init__()
        
        # 增加网络复杂度来处理相似的异常类型
        self.network = nn.Sequential(
            nn.Linear(11, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
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

def generate_extreme_challenging_test_data(n_samples=2000):
    """生成极具挑战性的测试数据，包含更多边界情况"""
    print(f"生成 {n_samples} 个极具挑战性的测试样本...")
    
    np.random.seed(456)  # 使用不同的随机种子
    
    test_data = []
    test_labels_binary = []
    test_labels_multiclass = []
    
    # 更复杂的测试样本类型
    sample_types = [
        'normal', 'borderline_normal', 'marginal_anomaly', 
        'noisy_normal', 'noisy_anomaly', 'mixed_signals',
        'extreme_borderline', 'conflicting_indicators', 'ambiguous_case'
    ]
    
    for i in range(n_samples):
        sample_type = np.random.choice(sample_types)
        
        if sample_type == 'normal':
            # 正常样本（30%）
            sample = [
                np.random.normal(75, 15),     # quality
                np.random.normal(-50, 10),    # signal
                np.random.normal(-90, 5),     # noise
                np.random.normal(15000, 5000), # rx_packets
                np.random.normal(12000, 4000), # tx_packets
                np.random.normal(3000000, 1000000), # rx_bytes
                np.random.normal(2500000, 800000),  # tx_bytes
                np.random.normal(20, 8),      # ping
                np.random.normal(30, 10),     # dns
                np.random.normal(40, 15),     # memory
                np.random.normal(25, 10)      # cpu
            ]
            binary_label = 0
            multiclass_label = 0
            
        elif sample_type == 'borderline_normal':
            # 边界正常样本（20%） - 接近异常阈值
            sample = [
                np.random.normal(60, 10),     # quality偏低
                np.random.normal(-65, 8),     # signal偏弱
                np.random.normal(-80, 6),     # noise偏高
                np.random.normal(8000, 2000), # packets偏少
                np.random.normal(6000, 1500), # packets偏少
                np.random.normal(1800000, 400000), # bytes偏少
                np.random.normal(1500000, 300000), # bytes偏少
                np.random.normal(40, 12),     # ping偏高
                np.random.normal(55, 15),     # dns偏高
                np.random.normal(65, 10),     # memory偏高
                np.random.normal(45, 8)       # cpu偏高
            ]
            binary_label = 0  # 仍然是正常，但很接近边界
            multiclass_label = 0
            
        elif sample_type == 'marginal_anomaly':
            # 边际异常样本（15%） - 刚超过正常范围
            base_patterns = ['wifi_degradation', 'network_latency', 'connection_instability']
            pattern = np.random.choice(base_patterns)
            
            if pattern == 'wifi_degradation':
                sample = [
                    np.random.normal(45, 15),     # quality差
                    np.random.normal(-70, 8),     # signal差
                    np.random.normal(-75, 8),     # noise高
                    np.random.normal(6000, 2000), # packets少
                    np.random.normal(4500, 1500), # packets少
                    np.random.normal(1200000, 400000), # bytes少
                    np.random.normal(1000000, 300000), # bytes少
                    np.random.normal(45, 15),     # ping稍高
                    np.random.normal(60, 20),     # dns稍高
                    np.random.normal(45, 12),     # memory正常
                    np.random.normal(30, 8)       # cpu正常
                ]
                multiclass_label = 1  # wifi_degradation
            elif pattern == 'network_latency':
                sample = [
                    np.random.normal(70, 12),     # quality正常
                    np.random.normal(-55, 8),     # signal正常
                    np.random.normal(-85, 5),     # noise正常
                    np.random.normal(12000, 3000), # packets正常
                    np.random.normal(9000, 2000), # packets正常
                    np.random.normal(2200000, 600000), # bytes正常
                    np.random.normal(1800000, 500000), # bytes正常
                    np.random.normal(65, 20),     # ping高
                    np.random.normal(90, 30),     # dns高
                    np.random.normal(40, 12),     # memory正常
                    np.random.normal(25, 8)       # cpu正常
                ]
                multiclass_label = 2  # network_latency
            else:  # connection_instability
                sample = [
                    np.random.normal(55, 20),     # quality不稳定
                    np.random.normal(-62, 12),    # signal不稳定
                    np.random.normal(-78, 10),    # noise较高
                    np.random.normal(7000, 3000), # packets减少
                    np.random.normal(5500, 2000), # packets减少
                    np.random.normal(1400000, 500000), # bytes减少
                    np.random.normal(1100000, 400000), # bytes减少
                    np.random.normal(50, 15),     # ping中等
                    np.random.normal(65, 20),     # dns中等
                    np.random.normal(38, 10),     # memory正常
                    np.random.normal(22, 6)       # cpu正常
                ]
                multiclass_label = 3  # connection_instability
            
            binary_label = 1
            
        elif sample_type in ['noisy_normal', 'noisy_anomaly']:
            # 带噪声的样本（15%）
            if sample_type == 'noisy_normal':
                base_sample = [75, -50, -90, 15000, 12000, 3000000, 2500000, 20, 30, 40, 25]
                binary_label = 0
                multiclass_label = 0
            else:
                base_sample = [45, -70, -75, 6000, 4500, 1200000, 1000000, 45, 60, 45, 30]
                binary_label = 1
                multiclass_label = 1
            
            sample = []
            for val in base_sample:
                # 添加20-40%的随机噪声
                noise = np.random.uniform(-0.4, 0.4) * abs(val) if val != 0 else np.random.uniform(-10, 10)
                sample.append(val + noise)
            
        elif sample_type == 'mixed_signals':
            # 混合信号样本（10%） - 部分指标正常，部分异常
            sample = [
                np.random.normal(75, 10),     # quality正常
                np.random.normal(-50, 8),     # signal正常
                np.random.normal(-90, 5),     # noise正常
                np.random.normal(3000, 1500), # packets异常低
                np.random.normal(2500, 1000), # packets异常低
                np.random.normal(600000, 300000), # bytes异常低
                np.random.normal(500000, 200000), # bytes异常低
                np.random.normal(25, 8),      # ping正常
                np.random.normal(35, 10),     # dns正常
                np.random.normal(40, 12),     # memory正常
                np.random.normal(25, 8)       # cpu正常
            ]
            binary_label = 1  # 应该被识别为异常
            multiclass_label = 3  # connection_instability
            
        elif sample_type == 'extreme_borderline':
            # 极端边界情况（5%） - 非常难以判断
            # 在正常和异常之间50:50混合
            normal_base = [75, -50, -90, 15000, 12000, 3000000, 2500000, 20, 30, 40, 25]
            anomaly_base = [50, -65, -80, 8000, 6000, 1500000, 1200000, 35, 50, 45, 30]
            
            sample = []
            for i, (normal_val, anomaly_val) in enumerate(zip(normal_base, anomaly_base)):
                # 50-50混合，然后添加噪声
                mix_factor = np.random.uniform(0.3, 0.7)
                mixed_val = normal_val * mix_factor + anomaly_val * (1 - mix_factor)
                noise = np.random.normal(0, 0.1 * abs(mixed_val))
                sample.append(mixed_val + noise)
            
            # 随机分配标签（边界情况很难确定）
            binary_label = np.random.choice([0, 1])
            multiclass_label = np.random.choice([0, 1, 2, 3]) if binary_label == 1 else 0
            
        elif sample_type == 'conflicting_indicators':
            # 冲突指标样本（3%） - 不同指标指向不同结论
            sample = [
                np.random.normal(80, 8),      # quality很好
                np.random.normal(-45, 5),     # signal很好
                np.random.normal(-92, 3),     # noise很低
                np.random.normal(500, 200),   # packets极少（异常）
                np.random.normal(400, 150),   # packets极少（异常）
                np.random.normal(100000, 50000), # bytes极少（异常）
                np.random.normal(80000, 30000),  # bytes极少（异常）
                np.random.normal(15, 5),      # ping很好
                np.random.normal(25, 5),      # dns很好
                np.random.normal(35, 8),      # memory正常
                np.random.normal(20, 5)       # cpu正常
            ]
            binary_label = 1  # 应该被识别为异常（流量异常）
            multiclass_label = 3  # connection_instability
            
        else:  # ambiguous_case
            # 模糊案例（2%） - 真正难以分类的情况
            # 所有值都在边界附近波动
            boundary_values = [62, -58, -82, 10000, 8000, 2000000, 1700000, 35, 45, 52, 38]
            sample = []
            for val in boundary_values:
                # 在边界值附近随机波动
                variation = np.random.normal(0, 0.15 * abs(val))
                sample.append(val + variation)
            
            # 真正随机的标签（模拟现实中的模糊情况）
            binary_label = np.random.choice([0, 1])
            multiclass_label = np.random.choice([0, 1, 2, 3]) if binary_label == 1 else 0
        
        # 应用边界约束
        sample[0] = np.clip(sample[0], 0, 100)    # quality
        sample[1] = np.clip(sample[1], -100, -10) # signal
        sample[2] = np.clip(sample[2], -100, -30) # noise
        for i in range(3, 7):  # packets and bytes
            sample[i] = max(0, sample[i])
        for i in range(7, 9):  # times
            sample[i] = max(1, sample[i])
        for i in range(9, 11): # percentages
            sample[i] = np.clip(sample[i], 0, 100)
        
        test_data.append(sample)
        test_labels_binary.append(binary_label)
        test_labels_multiclass.append(multiclass_label)
    
    return np.array(test_data), np.array(test_labels_binary), np.array(test_labels_multiclass)

def test_realistic_model_robustness():
    """测试真实数据模型的鲁棒性"""
    print("=== 真实数据模型鲁棒性测试 ===")
    print()
    
    # 加载真实数据模型和标准化器
    scaler = joblib.load('realistic_raw_data_scaler.pkl')
    
    detector_model = RealisticEndToEndAnomalyDetector()
    detector_model.load_state_dict(torch.load('realistic_end_to_end_anomaly_detector.pth', map_location='cpu'))
    detector_model.eval()
    
    classifier_model = RealisticEndToEndAnomalyClassifier(n_classes=6)
    classifier_model.load_state_dict(torch.load('realistic_end_to_end_anomaly_classifier.pth', map_location='cpu'))
    classifier_model.eval()
    
    # 生成极具挑战性的测试数据
    test_data, test_labels_binary, test_labels_multiclass = generate_extreme_challenging_test_data(2000)
    
    # 标准化测试数据
    test_data_scaled = scaler.transform(test_data)
    test_tensor = torch.FloatTensor(test_data_scaled)
    
    # 测试异常检测
    print("🔍 **异常检测性能测试（极端挑战）**")
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
    
    # 计算关键指标
    precision = cm_binary[1,1] / (cm_binary[1,1] + cm_binary[0,1]) if (cm_binary[1,1] + cm_binary[0,1]) > 0 else 0
    recall = cm_binary[1,1] / (cm_binary[1,1] + cm_binary[1,0]) if (cm_binary[1,1] + cm_binary[1,0]) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"   精确率: {precision:.3f}")
    print(f"   召回率: {recall:.3f}")
    print(f"   F1分数: {f1_score:.3f}")
    
    # 置信度分布
    confidence_normal = detection_probs[test_labels_binary == 0, 0].numpy()
    confidence_anomaly = detection_probs[test_labels_binary == 1, 1].numpy()
    
    print(f"   正常样本置信度: 均值={confidence_normal.mean():.3f}, 最小={confidence_normal.min():.3f}, 标准差={confidence_normal.std():.3f}")
    print(f"   异常样本置信度: 均值={confidence_anomaly.mean():.3f}, 最小={confidence_anomaly.min():.3f}, 标准差={confidence_anomaly.std():.3f}")
    
    # 测试异常分类（仅对检测为异常的样本）
    anomaly_mask = predicted_binary == 1
    if np.sum(anomaly_mask) > 0:
        print()
        print("🎯 **异常分类性能测试**")
        
        with torch.no_grad():
            classification_output = classifier_model(test_tensor[anomaly_mask])
            predicted_multiclass = torch.argmax(classification_output, dim=1).numpy()
        
        # 转换真实标签
        true_multiclass = test_labels_multiclass[anomaly_mask]
        
        # 只对真正的异常样本计算分类准确率
        true_anomaly_mask = test_labels_binary[anomaly_mask] == 1
        if np.sum(true_anomaly_mask) > 0:
            true_multiclass_filtered = true_multiclass[true_anomaly_mask] - 1
            predicted_multiclass_filtered = predicted_multiclass[true_anomaly_mask]
            
            # 确保标签在有效范围内
            valid_mask = (true_multiclass_filtered >= 0) & (true_multiclass_filtered < 6)
            if np.sum(valid_mask) > 0:
                classification_accuracy = accuracy_score(
                    true_multiclass_filtered[valid_mask], 
                    predicted_multiclass_filtered[valid_mask]
                )
                print(f"   异常分类准确率: {classification_accuracy:.3f} ({classification_accuracy*100:.1f}%)")
                print(f"   有效分类样本数: {np.sum(valid_mask)}")
            else:
                print("   没有有效的异常分类样本")
        else:
            print("   没有真正的异常样本被检测到")
    
    # 分析不同置信度阈值下的性能
    print()
    print("🔍 **不同置信度阈值下的性能**")
    thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]
    
    for threshold in thresholds:
        # 重新预测（使用置信度阈值）
        high_conf_mask = np.max(detection_probs.numpy(), axis=1) >= threshold
        
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = accuracy_score(
                test_labels_binary[high_conf_mask], 
                predicted_binary[high_conf_mask]
            )
            coverage = np.sum(high_conf_mask) / len(test_data)
            print(f"   置信度>={threshold}: 准确率={high_conf_accuracy:.3f}, 覆盖率={coverage:.3f}")
        else:
            print(f"   置信度>={threshold}: 无符合条件的样本")

def main():
    print("=== 真实数据模型极端鲁棒性测试 ===")
    print()
    
    print("🎯 **测试目标**:")
    print("   - 验证真实数据模型在极端情况下的表现")
    print("   - 测试边界情况和模糊样本的处理")
    print("   - 分析置信度分布和可靠性")
    print("   - 评估相比理想数据模型的改进")
    print()
    
    test_realistic_model_robustness()
    
    print()
    print("=== 测试分析结论 ===")
    print("预期表现: 相比理想数据模型，应该有:")
    print("1. 适度降低的准确率（更真实）")
    print("2. 更好的置信度分布（不再全是1.0）") 
    print("3. 更合理的错误模式")
    print("4. 更好的泛化能力")

if __name__ == "__main__":
    main() 