import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import subprocess

# 特征工程层：11维原始输入 → 6维特征
class FeatureEngineeringLayer(nn.Module):
    def __init__(self):
        super(FeatureEngineeringLayer, self).__init__()
        
    def forward(self, x):
        # 输入x: [batch_size, 11]
        # 按照README中的转换公式进行特征工程
        
        # 输入特征索引 (基于README中的11维原始数据)
        # 0: wlan0_wireless_quality, 1: wlan0_signal_level, 2: wlan0_noise_level
        # 3: wlan0_rx_packets, 4: wlan0_tx_packets, 5: wlan0_rx_bytes, 6: wlan0_tx_bytes  
        # 7: gateway_ping_time, 8: dns_resolution_time
        # 9: memory_usage_percent, 10: cpu_usage_percent
        
        # 1. 平均信号强度 = (信号质量 + |信号强度|) / 20
        avg_signal_strength = (x[:, 0] + torch.abs(x[:, 1])) / 20.0
        
        # 2. 平均数据传输率 = min((接收字节数 + 发送字节数) / 5000000, 1.0)
        avg_data_rate = torch.clamp((x[:, 5] + x[:, 6]) / 5000000.0, max=1.0)
        
        # 3. 平均网络延迟 = (网关ping + DNS解析时间) / 2
        avg_latency = (x[:, 7] + x[:, 8]) / 2.0
        
        # 4. 丢包率估算 = max(0, (|噪声水平| - 70) / 200)
        packet_loss_rate = torch.clamp((torch.abs(x[:, 2]) - 70) / 200.0, min=0.0)
        
        # 5. 系统负载 = (CPU使用率 + 内存使用率) / 200
        system_load = (x[:, 10] + x[:, 9]) / 200.0
        
        # 6. 网络稳定性 = min((接收包数 + 发送包数) / 50000, 1.0)
        network_stability = torch.clamp((x[:, 3] + x[:, 4]) / 50000.0, max=1.0)
        
        # 堆叠为6维特征向量
        features = torch.stack([
            avg_signal_strength,
            avg_data_rate, 
            avg_latency,
            packet_loss_rate,
            system_load,
            network_stability
        ], dim=1)
        
        return features

# 端到端异常检测网络 (11维输入)
class EndToEndAnomalyDetector(nn.Module):
    def __init__(self):
        super(EndToEndAnomalyDetector, self).__init__()
        
        # 特征工程层: 11维 → 6维
        self.feature_engineering = FeatureEngineeringLayer()
        
        # 异常检测网络: 6维 → 2分类
        self.detector = nn.Sequential(
            nn.Linear(6, 64),
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
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, 11] 原始网络监控数据
        features = self.feature_engineering(x)  # [batch_size, 6]
        detection_scores = self.detector(features)  # [batch_size, 2]
        return detection_scores

# 端到端异常分类网络 (11维输入)
class EndToEndAnomalyClassifier(nn.Module):
    def __init__(self, n_classes=6):
        super(EndToEndAnomalyClassifier, self).__init__()
        
        # 特征工程层: 11维 → 6维
        self.feature_engineering = FeatureEngineeringLayer()
        
        # 异常分类网络: 6维 → 6分类
        self.classifier = nn.Sequential(
            nn.Linear(6, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, n_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, 11] 原始网络监控数据
        features = self.feature_engineering(x)  # [batch_size, 6]
        classification_scores = self.classifier(features)  # [batch_size, 6]
        return classification_scores

def convert_to_onnx(model, input_shape, output_path):
    """将PyTorch模型转换为ONNX格式"""
    print(f"转换模型为ONNX格式: {output_path}")
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, *input_shape)
    
    # 转换为ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
    
    print(f"✅ ONNX模型已保存: {output_path}")

def convert_to_dlc(onnx_path, dlc_path):
    """将ONNX模型转换为DLC格式"""
    print(f"转换ONNX为DLC格式: {dlc_path}")
    
    # 设置命令参数
    cmd = [
        './2.26.2.240911/bin/x86_64-linux-clang/snpe-onnx-to-dlc',
        '--input_network', onnx_path,
        '--output_path', dlc_path,
        '--input_dim', 'input', '11'
    ]
    
    try:
        # 执行转换
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ DLC模型已生成: {dlc_path}")
        
        # 显示文件大小
        if os.path.exists(dlc_path):
            file_size = os.path.getsize(dlc_path)
            print(f"   文件大小: {file_size / 1024:.1f} KB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ DLC转换失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    print("=== 端到端模型转换为DLC格式 ===")
    print()
    
    # 1. 加载训练好的端到端模型
    print("🔄 加载训练好的端到端模型...")
    
    # 异常检测模型
    detector_model = EndToEndAnomalyDetector()
    detector_model.load_state_dict(torch.load('end_to_end_anomaly_detector.pth', map_location='cpu'))
    detector_model.eval()
    
    # 异常分类模型
    classifier_model = EndToEndAnomalyClassifier(n_classes=6)
    classifier_model.load_state_dict(torch.load('end_to_end_anomaly_classifier.pth', map_location='cpu'))
    classifier_model.eval()
    
    print("✅ 模型加载完成")
    
    # 2. 转换为ONNX格式
    print("\n🔄 转换为ONNX格式...")
    
    # 异常检测模型 -> ONNX
    convert_to_onnx(detector_model, (11,), 'end_to_end_anomaly_detector.onnx')
    
    # 异常分类模型 -> ONNX
    convert_to_onnx(classifier_model, (11,), 'end_to_end_anomaly_classifier.onnx')
    
    # 3. 转换为DLC格式
    print("\n🔄 转换为DLC格式...")
    
    # 异常检测模型 -> DLC
    detector_success = convert_to_dlc('end_to_end_anomaly_detector.onnx', 'end_to_end_anomaly_detector.dlc')
    
    # 异常分类模型 -> DLC
    classifier_success = convert_to_dlc('end_to_end_anomaly_classifier.onnx', 'end_to_end_anomaly_classifier.dlc')
    
    # 4. 结果汇总
    print("\n=== 转换结果汇总 ===")
    
    if detector_success and classifier_success:
        print("✅ 所有模型转换成功!")
        print()
        print("📁 生成的DLC文件:")
        print("   - end_to_end_anomaly_detector.dlc    (异常检测)")
        print("   - end_to_end_anomaly_classifier.dlc  (异常分类)")
        print()
        print("🎯 **端到端方案优势**:")
        print("   ✅ 直接处理11维原始网络监控数据")
        print("   ✅ 内置特征工程转换（11维 → 6维）")
        print("   ✅ 移动设备无需额外预处理代码")
        print("   ✅ 完整的两阶段异常检测流程")
        print("   ✅ 真正的端到端部署解决方案")
        
        # 显示总文件大小
        total_size = 0
        for filename in ['end_to_end_anomaly_detector.dlc', 'end_to_end_anomaly_classifier.dlc']:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                total_size += size
        
        print(f"   📦 总DLC文件大小: {total_size / 1024:.1f} KB")
        
    else:
        print("❌ 部分模型转换失败")
        if not detector_success:
            print("   - 异常检测模型转换失败")
        if not classifier_success:
            print("   - 异常分类模型转换失败")

if __name__ == "__main__":
    main() 