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

def generate_realistic_network_data(n_samples=20000):
    """生成更真实的网络监控数据，包含重叠区域和渐变异常"""
    print(f"生成 {n_samples} 个真实的网络监控样本...")
    
    np.random.seed(42)
    
    # 定义正常网络的基准值和变化范围
    normal_baseline = {
        'wlan0_wireless_quality': (75, 15),     # (均值, 标准差)
        'wlan0_signal_level': (-50, 10),        # dBm
        'wlan0_noise_level': (-90, 5),          # dBm
        'wlan0_rx_packets': (15000, 5000),      # 包数
        'wlan0_tx_packets': (12000, 4000),      # 包数
        'wlan0_rx_bytes': (3000000, 1000000),   # 字节
        'wlan0_tx_bytes': (2500000, 800000),    # 字节
        'gateway_ping_time': (20, 8),           # 毫秒
        'dns_resolution_time': (30, 10),        # 毫秒
        'memory_usage_percent': (40, 15),       # 百分比
        'cpu_usage_percent': (25, 10)           # 百分比
    }
    
    # 正常样本 (75% - 更接近实际比例)
    n_normal = int(n_samples * 0.75)
    normal_data = []
    
    for _ in range(n_normal):
        sample = []
        for feature, (mean, std) in normal_baseline.items():
            # 使用正态分布 + 适当的边界
            value = np.random.normal(mean, std)
            
            # 添加合理的边界约束
            if 'quality' in feature:
                value = np.clip(value, 20, 100)  # 信号质量范围
            elif 'signal_level' in feature:
                value = np.clip(value, -80, -20)  # 信号强度范围
            elif 'noise_level' in feature:
                value = np.clip(value, -100, -70)  # 噪声范围
            elif 'packets' in feature:
                value = max(0, value)  # 包数不能为负
            elif 'bytes' in feature:
                value = max(0, value)  # 字节数不能为负
            elif 'time' in feature:
                value = max(1, value)  # 时间不能为0或负数
            elif 'percent' in feature:
                value = np.clip(value, 0, 100)  # 百分比范围
            
            sample.append(value)
        
        normal_data.append(sample)
    
    # 异常样本 (25%)
    n_anomaly = n_samples - n_normal
    anomaly_data = []
    anomaly_labels = []
    
    # 定义更真实的异常模式（与正常状态有部分重叠）
    anomaly_patterns = {
        'wifi_degradation': {  # WiFi信号衰减（轻微异常）
            'wlan0_wireless_quality': (50, 20),     # 质量下降但不极端
            'wlan0_signal_level': (-65, 10),        # 信号稍弱
            'wlan0_noise_level': (-80, 8),          # 噪声稍高
            'wlan0_rx_packets': (8000, 3000),       # 包数减少
            'wlan0_tx_packets': (6000, 2500),       # 包数减少
            'wlan0_rx_bytes': (1500000, 600000),    # 流量减少
            'wlan0_tx_bytes': (1200000, 500000),    # 流量减少
            'gateway_ping_time': (35, 15),          # ping稍长
            'dns_resolution_time': (50, 20),        # DNS稍慢
            'memory_usage_percent': (45, 15),       # 内存正常
            'cpu_usage_percent': (30, 12)           # CPU正常
        },
        'network_latency': {  # 网络延迟（中等异常）
            'wlan0_wireless_quality': (70, 20),     # 质量尚可
            'wlan0_signal_level': (-55, 12),        # 信号正常
            'wlan0_noise_level': (-85, 8),          # 噪声正常
            'wlan0_rx_packets': (12000, 4000),      # 包数正常
            'wlan0_tx_packets': (10000, 3000),      # 包数正常
            'wlan0_rx_bytes': (2500000, 800000),    # 流量正常
            'wlan0_tx_bytes': (2000000, 600000),    # 流量正常
            'gateway_ping_time': (80, 25),          # ping明显长
            'dns_resolution_time': (120, 40),       # DNS明显慢
            'memory_usage_percent': (40, 15),       # 内存正常
            'cpu_usage_percent': (25, 10)           # CPU正常
        },
        'connection_instability': {  # 连接不稳定（数据包问题）
            'wlan0_wireless_quality': (60, 25),     # 质量不稳定
            'wlan0_signal_level': (-60, 15),        # 信号不稳定
            'wlan0_noise_level': (-75, 12),         # 噪声较高
            'wlan0_rx_packets': (5000, 3000),       # 包数明显少
            'wlan0_tx_packets': (4000, 2500),       # 包数明显少
            'wlan0_rx_bytes': (800000, 500000),     # 流量少
            'wlan0_tx_bytes': (600000, 400000),     # 流量少
            'gateway_ping_time': (45, 20),          # ping中等
            'dns_resolution_time': (60, 25),        # DNS中等
            'memory_usage_percent': (35, 15),       # 内存正常
            'cpu_usage_percent': (20, 8)            # CPU正常
        },
        'bandwidth_congestion': {  # 带宽拥塞（高流量异常）
            'wlan0_wireless_quality': (80, 15),     # 质量好
            'wlan0_signal_level': (-45, 8),         # 信号好
            'wlan0_noise_level': (-90, 5),          # 噪声低
            'wlan0_rx_packets': (25000, 8000),      # 包数很多
            'wlan0_tx_packets': (20000, 6000),      # 包数很多
            'wlan0_rx_bytes': (8000000, 2000000),   # 流量很高
            'wlan0_tx_bytes': (6500000, 1500000),   # 流量很高
            'gateway_ping_time': (50, 20),          # ping稍长
            'dns_resolution_time': (40, 15),        # DNS正常
            'memory_usage_percent': (60, 20),       # 内存使用高
            'cpu_usage_percent': (45, 15)           # CPU使用高
        },
        'system_stress': {  # 系统压力（资源异常）
            'wlan0_wireless_quality': (75, 15),     # 质量正常
            'wlan0_signal_level': (-50, 10),        # 信号正常
            'wlan0_noise_level': (-90, 5),          # 噪声正常
            'wlan0_rx_packets': (14000, 4000),      # 包数正常
            'wlan0_tx_packets': (11000, 3000),      # 包数正常
            'wlan0_rx_bytes': (2800000, 800000),    # 流量正常
            'wlan0_tx_bytes': (2300000, 700000),    # 流量正常
            'gateway_ping_time': (30, 12),          # ping正常
            'dns_resolution_time': (40, 15),        # DNS正常
            'memory_usage_percent': (85, 10),       # 内存很高
            'cpu_usage_percent': (80, 15)           # CPU很高
        },
        'dns_issues': {  # DNS问题（服务异常）
            'wlan0_wireless_quality': (75, 15),     # 质量正常
            'wlan0_signal_level': (-50, 10),        # 信号正常
            'wlan0_noise_level': (-90, 5),          # 噪声正常
            'wlan0_rx_packets': (15000, 4000),      # 包数正常
            'wlan0_tx_packets': (12000, 3000),      # 包数正常
            'wlan0_rx_bytes': (3000000, 800000),    # 流量正常
            'wlan0_tx_bytes': (2500000, 600000),    # 流量正常
            'gateway_ping_time': (25, 10),          # ping正常
            'dns_resolution_time': (200, 80),       # DNS很慢
            'memory_usage_percent': (40, 15),       # 内存正常
            'cpu_usage_percent': (25, 10)           # CPU正常
        }
    }
    
    pattern_names = list(anomaly_patterns.keys())
    
    for _ in range(n_anomaly):
        # 随机选择异常类型
        pattern_name = np.random.choice(pattern_names)
        pattern = anomaly_patterns[pattern_name]
        
        sample = []
        for i, (feature, (mean, std)) in enumerate(pattern.items()):
            # 生成异常值，但添加一些随机性和边界情况
            value = np.random.normal(mean, std)
            
            # 10%的概率产生边界情况（接近正常值）
            if np.random.random() < 0.1:
                normal_mean, normal_std = list(normal_baseline.values())[i]
                # 在正常值和异常值之间插值
                interpolation_factor = np.random.uniform(0.3, 0.7)
                value = value * interpolation_factor + normal_mean * (1 - interpolation_factor)
            
            # 添加5%的噪声
            if np.random.random() < 0.05:
                noise_factor = np.random.uniform(0.8, 1.2)
                value *= noise_factor
            
            # 应用边界约束
            if 'quality' in feature:
                value = np.clip(value, 0, 100)
            elif 'signal_level' in feature:
                value = np.clip(value, -100, -10)
            elif 'noise_level' in feature:
                value = np.clip(value, -100, -30)
            elif 'packets' in feature:
                value = max(0, value)
            elif 'bytes' in feature:
                value = max(0, value)
            elif 'time' in feature:
                value = max(1, value)
            elif 'percent' in feature:
                value = np.clip(value, 0, 100)
            
            sample.append(value)
        
        anomaly_data.append(sample)
        anomaly_labels.append(pattern_names.index(pattern_name))
    
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
    print(f"  异常类型分布: {np.bincount(anomaly_labels)}")
    
    return X, y_binary, y_multiclass

def train_model(model, train_loader, val_loader, epochs=150, lr=0.001):
    """训练模型，使用更严格的训练策略"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 25  # 增加耐心，避免过早停止
    
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
            
            # 梯度裁剪，提高训练稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        if epoch % 20 == 0:
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
    print("=== 真实数据端到端神经网络训练（11维原始输入）===")
    print()
    
    # 1. 生成真实的11维原始网络监控数据
    X, y_binary, y_multiclass = generate_realistic_network_data(n_samples=30000)  # 增加样本数
    
    # 2. 标准化11维原始数据
    print("对11维原始数据进行标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 保存标准化器
    import joblib
    joblib.dump(scaler, 'realistic_raw_data_scaler.pkl')
    print("标准化器已保存为 'realistic_raw_data_scaler.pkl'")
    
    # 3. 训练异常检测网络（二分类）
    print("\n=== 训练真实异常检测网络（11维输入 → 2分类）===")
    
    # 准备异常检测数据
    X_det_train, X_det_test, y_det_train, y_det_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # 转换为张量
    X_det_train_tensor = torch.FloatTensor(X_det_train)
    y_det_train_tensor = torch.LongTensor(y_det_train)
    X_det_test_tensor = torch.FloatTensor(X_det_test)
    y_det_test_tensor = torch.LongTensor(y_det_test)
    
    # 创建数据加载器，使用更小的batch size来提高训练稳定性
    train_det_dataset = TensorDataset(X_det_train_tensor, y_det_train_tensor)
    train_det_loader = DataLoader(train_det_dataset, batch_size=32, shuffle=True)
    
    val_det_dataset = TensorDataset(X_det_test_tensor, y_det_test_tensor)
    val_det_loader = DataLoader(val_det_dataset, batch_size=32, shuffle=False)
    
    # 创建并训练异常检测模型
    detector_model = RealisticEndToEndAnomalyDetector()
    print(f"真实异常检测网络参数数量: {sum(p.numel() for p in detector_model.parameters()):,}")
    
    train_losses, val_losses, val_accuracies, best_val_acc = train_model(
        detector_model, train_det_loader, val_det_loader, epochs=150, lr=0.001
    )
    
    print(f"真实异常检测网络最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存异常检测模型
    torch.save(detector_model.state_dict(), 'realistic_end_to_end_anomaly_detector.pth')
    print("真实异常检测模型已保存为 'realistic_end_to_end_anomaly_detector.pth'")
    
    # 4. 训练异常分类网络（仅异常样本）
    print("\n=== 训练真实异常分类网络（11维输入 → 6分类）===")
    
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
    train_cls_loader = DataLoader(train_cls_dataset, batch_size=16, shuffle=True)  # 更小的batch size
    
    val_cls_dataset = TensorDataset(X_cls_test_tensor, y_cls_test_tensor)
    val_cls_loader = DataLoader(val_cls_dataset, batch_size=16, shuffle=False)
    
    # 创建并训练异常分类模型
    classifier_model = RealisticEndToEndAnomalyClassifier(n_classes=6)
    print(f"真实异常分类网络参数数量: {sum(p.numel() for p in classifier_model.parameters()):,}")
    
    train_losses, val_losses, val_accuracies, best_val_acc = train_model(
        classifier_model, train_cls_loader, val_cls_loader, epochs=150, lr=0.0005  # 更小的学习率
    )
    
    print(f"真实异常分类网络最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存异常分类模型
    torch.save(classifier_model.state_dict(), 'realistic_end_to_end_anomaly_classifier.pth')
    print("真实异常分类模型已保存为 'realistic_end_to_end_anomaly_classifier.pth'")
    
    print("\n=== 真实数据训练完成 ===")
    print("✅ 两个真实端到端模型已成功训练并保存")
    print("✅ 使用更真实的数据分布和重叠区域")
    print("✅ 增加了边界情况和噪声处理")
    print("✅ 更接近实际网络监控场景")

if __name__ == "__main__":
    main() 