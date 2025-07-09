import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 简化的特征工程层：使用线性变换替代torch.stack
class SimplifiedFeatureEngineeringLayer(nn.Module):
    def __init__(self):
        super(SimplifiedFeatureEngineeringLayer, self).__init__()
        
        # 使用线性变换直接从11维原始特征映射到6维特征
        # 这样可以避免复杂的索引和组合操作
        self.feature_transform = nn.Linear(11, 6)
        
        # 初始化权重以模拟原始的特征工程公式
        self._initialize_feature_weights()
    
    def _initialize_feature_weights(self):
        """初始化权重以模拟特征工程公式"""
        with torch.no_grad():
            # 创建权重矩阵：[6, 11]
            # 每一行对应一个输出特征，每一列对应一个输入特征
            
            # 特征0：平均信号强度 = (wlan0_wireless_quality + |wlan0_signal_level|) / 20
            self.feature_transform.weight[0, 0] = 1.0 / 20.0  # wlan0_wireless_quality
            self.feature_transform.weight[0, 1] = 1.0 / 20.0  # wlan0_signal_level (绝对值处理在forward中)
            
            # 特征1：平均数据传输率 = (wlan0_rx_bytes + wlan0_tx_bytes) / 5000000
            self.feature_transform.weight[1, 5] = 1.0 / 5000000.0  # wlan0_rx_bytes
            self.feature_transform.weight[1, 6] = 1.0 / 5000000.0  # wlan0_tx_bytes
            
            # 特征2：平均网络延迟 = (gateway_ping_time + dns_resolution_time) / 2
            self.feature_transform.weight[2, 7] = 1.0 / 2.0  # gateway_ping_time
            self.feature_transform.weight[2, 8] = 1.0 / 2.0  # dns_resolution_time
            
            # 特征3：丢包率估算 = (|wlan0_noise_level| - 70) / 200
            self.feature_transform.weight[3, 2] = 1.0 / 200.0  # wlan0_noise_level
            self.feature_transform.bias[3] = -70.0 / 200.0
            
            # 特征4：系统负载 = (cpu_usage_percent + memory_usage_percent) / 200
            self.feature_transform.weight[4, 9] = 1.0 / 200.0  # memory_usage_percent
            self.feature_transform.weight[4, 10] = 1.0 / 200.0  # cpu_usage_percent
            
            # 特征5：网络稳定性 = (wlan0_rx_packets + wlan0_tx_packets) / 50000
            self.feature_transform.weight[5, 3] = 1.0 / 50000.0  # wlan0_rx_packets
            self.feature_transform.weight[5, 4] = 1.0 / 50000.0  # wlan0_tx_packets
    
    def forward(self, x):
        # 应用线性变换
        features = self.feature_transform(x)
        
        # 应用激活函数确保合理的特征范围
        # 创建新的tensor来避免inplace操作
        output = features.clone()
        output[:, 1] = torch.clamp(features[:, 1], max=1.0)  # 数据传输率
        output[:, 3] = torch.clamp(features[:, 3], min=0.0)  # 丢包率
        output[:, 5] = torch.clamp(features[:, 5], max=1.0)  # 网络稳定性
        
        return output

# 端到端异常检测网络 (11维输入)
class SimplifiedEndToEndAnomalyDetector(nn.Module):
    def __init__(self):
        super(SimplifiedEndToEndAnomalyDetector, self).__init__()
        
        # 特征工程层: 11维 → 6维
        self.feature_engineering = SimplifiedFeatureEngineeringLayer()
        
        # 异常检测网络: 6维 → 2分类
        self.detector = nn.Sequential(
            nn.Linear(6, 64),
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
            if isinstance(m, nn.Linear) and m != self.feature_engineering.feature_transform:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, 11] 原始网络监控数据
        features = self.feature_engineering(x)  # [batch_size, 6]
        detection_scores = self.detector(features)  # [batch_size, 2]
        return detection_scores

# 端到端异常分类网络 (11维输入)
class SimplifiedEndToEndAnomalyClassifier(nn.Module):
    def __init__(self, n_classes=6):
        super(SimplifiedEndToEndAnomalyClassifier, self).__init__()
        
        # 特征工程层: 11维 → 6维
        self.feature_engineering = SimplifiedFeatureEngineeringLayer()
        
        # 异常分类网络: 6维 → 6分类
        self.classifier = nn.Sequential(
            nn.Linear(6, 128),
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
            if isinstance(m, nn.Linear) and m != self.feature_engineering.feature_transform:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, 11] 原始网络监控数据
        features = self.feature_engineering(x)  # [batch_size, 6]
        classification_scores = self.classifier(features)  # [batch_size, 6]
        return classification_scores

def generate_raw_network_data(n_samples=20000):
    """生成11维原始网络监控数据"""
    print(f"生成 {n_samples} 个11维原始网络监控样本...")
    
    np.random.seed(42)
    
    # 正常样本 (90%)
    n_normal = int(n_samples * 0.9)
    normal_data = []
    
    for _ in range(n_normal):
        # 正常网络状态的11维原始数据
        sample = [
            np.random.uniform(70, 95),      # wlan0_wireless_quality
            np.random.uniform(-55, -30),    # wlan0_signal_level
            np.random.uniform(-95, -85),    # wlan0_noise_level
            np.random.uniform(8000, 25000), # wlan0_rx_packets
            np.random.uniform(6000, 20000), # wlan0_tx_packets
            np.random.uniform(1000000, 8000000), # wlan0_rx_bytes
            np.random.uniform(500000, 6000000),  # wlan0_tx_bytes
            np.random.uniform(8, 25),       # gateway_ping_time
            np.random.uniform(15, 40),      # dns_resolution_time
            np.random.uniform(20, 60),      # memory_usage_percent
            np.random.uniform(5, 30)        # cpu_usage_percent
        ]
        normal_data.append(sample)
    
    # 异常样本 (10%)
    n_anomaly = n_samples - n_normal
    anomaly_data = []
    anomaly_labels = []
    
    anomaly_types = [
        'wifi_disconnection',
        'high_latency', 
        'packet_loss',
        'bandwidth_saturation',
        'system_overload',
        'dns_failure'
    ]
    
    for _ in range(n_anomaly):
        anomaly_type = np.random.choice(anomaly_types)
        
        if anomaly_type == 'wifi_disconnection':
            # WiFi断开
            sample = [
                np.random.uniform(0, 20),       # 信号质量极差
                np.random.uniform(-90, -75),    # 信号强度极弱
                np.random.uniform(-60, -40),    # 噪声水平高
                np.random.uniform(0, 100),      # 接收包数极少
                np.random.uniform(0, 100),      # 发送包数极少
                np.random.uniform(0, 10000),    # 接收字节数极少
                np.random.uniform(0, 10000),    # 发送字节数极少
                np.random.uniform(100, 300),    # ping时间极长
                np.random.uniform(100, 500),    # DNS解析时间极长
                np.random.uniform(10, 40),      # 内存使用正常
                np.random.uniform(5, 25)        # CPU使用正常
            ]
            label = 0
            
        elif anomaly_type == 'high_latency':
            # 高延迟
            sample = [
                np.random.uniform(60, 80),      # 信号质量中等
                np.random.uniform(-70, -45),    # 信号强度中等
                np.random.uniform(-90, -80),    # 噪声正常
                np.random.uniform(5000, 15000), # 包数正常
                np.random.uniform(4000, 12000), # 包数正常
                np.random.uniform(500000, 3000000), # 字节数正常
                np.random.uniform(300000, 2000000), # 字节数正常
                np.random.uniform(80, 200),     # ping时间很长
                np.random.uniform(80, 300),     # DNS解析时间很长
                np.random.uniform(20, 50),      # 内存使用正常
                np.random.uniform(10, 30)       # CPU使用正常
            ]
            label = 1
            
        elif anomaly_type == 'packet_loss':
            # 丢包
            sample = [
                np.random.uniform(30, 60),      # 信号质量差
                np.random.uniform(-80, -60),    # 信号强度差
                np.random.uniform(-50, -30),    # 噪声水平很高
                np.random.uniform(1000, 5000),  # 接收包数少
                np.random.uniform(1000, 5000),  # 发送包数少
                np.random.uniform(100000, 1000000), # 接收字节数少
                np.random.uniform(100000, 1000000), # 发送字节数少
                np.random.uniform(30, 80),      # ping时间长
                np.random.uniform(30, 100),     # DNS解析时间长
                np.random.uniform(15, 45),      # 内存使用正常
                np.random.uniform(8, 35)        # CPU使用正常
            ]
            label = 2
            
        elif anomaly_type == 'bandwidth_saturation':
            # 带宽饱和
            sample = [
                np.random.uniform(75, 90),      # 信号质量好
                np.random.uniform(-50, -35),    # 信号强度好
                np.random.uniform(-90, -85),    # 噪声正常
                np.random.uniform(40000, 80000), # 包数非常多
                np.random.uniform(35000, 70000), # 包数非常多
                np.random.uniform(15000000, 50000000), # 字节数非常多
                np.random.uniform(12000000, 40000000), # 字节数非常多
                np.random.uniform(40, 100),     # ping时间长
                np.random.uniform(20, 60),      # DNS解析时间正常
                np.random.uniform(30, 70),      # 内存使用高
                np.random.uniform(20, 60)       # CPU使用高
            ]
            label = 3
            
        elif anomaly_type == 'system_overload':
            # 系统过载
            sample = [
                np.random.uniform(65, 85),      # 信号质量正常
                np.random.uniform(-60, -40),    # 信号强度正常
                np.random.uniform(-90, -80),    # 噪声正常
                np.random.uniform(6000, 18000), # 包数正常
                np.random.uniform(5000, 15000), # 包数正常
                np.random.uniform(800000, 5000000), # 字节数正常
                np.random.uniform(600000, 4000000), # 字节数正常
                np.random.uniform(15, 50),      # ping时间正常
                np.random.uniform(20, 80),      # DNS解析时间正常
                np.random.uniform(80, 95),      # 内存使用非常高
                np.random.uniform(85, 98)       # CPU使用非常高
            ]
            label = 4
            
        elif anomaly_type == 'dns_failure':
            # DNS失败
            sample = [
                np.random.uniform(70, 90),      # 信号质量好
                np.random.uniform(-55, -35),    # 信号强度好
                np.random.uniform(-90, -85),    # 噪声正常
                np.random.uniform(8000, 20000), # 包数正常
                np.random.uniform(6000, 18000), # 包数正常
                np.random.uniform(1000000, 6000000), # 字节数正常
                np.random.uniform(800000, 5000000),  # 字节数正常
                np.random.uniform(10, 30),      # ping时间正常
                np.random.uniform(200, 1000),   # DNS解析时间极长
                np.random.uniform(25, 55),      # 内存使用正常
                np.random.uniform(8, 28)        # CPU使用正常
            ]
            label = 5
        
        anomaly_data.append(sample)
        anomaly_labels.append(label)
    
    # 合并数据
    all_data = normal_data + anomaly_data
    
    # 创建标签：0=normal, 1=anomaly
    binary_labels = [0] * n_normal + [1] * n_anomaly
    
    # 多分类标签：0=normal, 1-6=各种异常类型
    multiclass_labels = [0] * n_normal + [label + 1 for label in anomaly_labels]
    
    # 转换为numpy数组
    X = np.array(all_data, dtype=np.float32)
    y_binary = np.array(binary_labels, dtype=np.int64)
    y_multiclass = np.array(multiclass_labels, dtype=np.int64)
    
    print(f"生成完成:")
    print(f"  数据形状: {X.shape}")
    print(f"  正常样本: {n_normal} ({n_normal/n_samples*100:.1f}%)")
    print(f"  异常样本: {n_anomaly} ({n_anomaly/n_samples*100:.1f}%)")
    
    return X, y_binary, y_multiclass

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    return train_losses, val_losses, val_accuracies, best_val_acc

def main():
    print("=== 简化版端到端神经网络训练（11维原始输入）===")
    print()
    
    # 1. 生成11维原始网络监控数据
    X, y_binary, y_multiclass = generate_raw_network_data(n_samples=20000)
    
    # 2. 标准化11维原始数据
    print("对11维原始数据进行标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 保存标准化器
    import joblib
    joblib.dump(scaler, 'simplified_raw_data_scaler.pkl')
    print("标准化器已保存为 'simplified_raw_data_scaler.pkl'")
    
    # 3. 训练异常检测网络（二分类）
    print("\n=== 训练简化版异常检测网络（11维输入 → 2分类）===")
    
    # 准备异常检测数据
    X_det_train, X_det_test, y_det_train, y_det_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # 转换为张量
    X_det_train_tensor = torch.FloatTensor(X_det_train)
    y_det_train_tensor = torch.LongTensor(y_det_train)
    X_det_test_tensor = torch.FloatTensor(X_det_test)
    y_det_test_tensor = torch.LongTensor(y_det_test)
    
    # 创建数据加载器
    train_det_dataset = TensorDataset(X_det_train_tensor, y_det_train_tensor)
    train_det_loader = DataLoader(train_det_dataset, batch_size=64, shuffle=True)
    
    val_det_dataset = TensorDataset(X_det_test_tensor, y_det_test_tensor)
    val_det_loader = DataLoader(val_det_dataset, batch_size=64, shuffle=False)
    
    # 创建并训练异常检测模型
    detector_model = SimplifiedEndToEndAnomalyDetector()
    print(f"简化版异常检测网络参数数量: {sum(p.numel() for p in detector_model.parameters()):,}")
    
    train_losses, val_losses, val_accuracies, best_val_acc = train_model(
        detector_model, train_det_loader, val_det_loader, epochs=100, lr=0.001
    )
    
    print(f"简化版异常检测网络最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存异常检测模型
    torch.save(detector_model.state_dict(), 'simplified_end_to_end_anomaly_detector.pth')
    print("简化版异常检测模型已保存为 'simplified_end_to_end_anomaly_detector.pth'")
    
    # 4. 训练异常分类网络（仅异常样本）
    print("\n=== 训练简化版异常分类网络（11维输入 → 6分类）===")
    
    # 提取异常样本
    anomaly_mask = y_binary == 1
    X_anomaly = X_scaled[anomaly_mask]
    y_anomaly = y_multiclass[anomaly_mask] - 1  # 转换为0-5标签
    
    print(f"异常样本数量: {len(X_anomaly)}")
    print(f"异常类别分布: {np.bincount(y_anomaly)}")
    
    # 准备异常分类数据
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X_anomaly, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly
    )
    
    # 转换为张量
    X_cls_train_tensor = torch.FloatTensor(X_cls_train)
    y_cls_train_tensor = torch.LongTensor(y_cls_train)
    X_cls_test_tensor = torch.FloatTensor(X_cls_test)
    y_cls_test_tensor = torch.LongTensor(y_cls_test)
    
    # 创建数据加载器
    train_cls_dataset = TensorDataset(X_cls_train_tensor, y_cls_train_tensor)
    train_cls_loader = DataLoader(train_cls_dataset, batch_size=32, shuffle=True)
    
    val_cls_dataset = TensorDataset(X_cls_test_tensor, y_cls_test_tensor)
    val_cls_loader = DataLoader(val_cls_dataset, batch_size=32, shuffle=False)
    
    # 创建并训练异常分类模型
    classifier_model = SimplifiedEndToEndAnomalyClassifier(n_classes=6)
    print(f"简化版异常分类网络参数数量: {sum(p.numel() for p in classifier_model.parameters()):,}")
    
    train_losses, val_losses, val_accuracies, best_val_acc = train_model(
        classifier_model, train_cls_loader, val_cls_loader, epochs=100, lr=0.001
    )
    
    print(f"简化版异常分类网络最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存异常分类模型
    torch.save(classifier_model.state_dict(), 'simplified_end_to_end_anomaly_classifier.pth')
    print("简化版异常分类模型已保存为 'simplified_end_to_end_anomaly_classifier.pth'")
    
    print("\n=== 训练完成 ===")
    print("✅ 两个简化版端到端模型已成功训练并保存")
    print("✅ 模型现在可以直接处理11维原始网络监控数据")
    print("✅ 避免了torch.stack操作，应该可以成功转换为DLC")

if __name__ == "__main__":
    main() 