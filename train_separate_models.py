#!/usr/bin/env python3
"""
分别训练异常检测模型和异常分类模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# 异常检测模型（二分类）
class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.layers(x)

# 异常分类模型（六分类）
class AnomalyClassifier(nn.Module):
    def __init__(self):
        super(AnomalyClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 6)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.layers(x)

def generate_data():
    print("🔄 Generating data...")
    from train_realistic_end_to_end_networks import generate_realistic_network_data
    
    X, y_binary, y_multiclass = generate_realistic_network_data(n_samples=30000)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'separate_models_scaler.pkl')
    print("✅ Scaler saved to separate_models_scaler.pkl")
    
    return X_scaled, y_binary, y_multiclass

def train_detector(X, y_binary):
    print("\n🚀 Training Anomaly Detector...")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 训练模型
    model = AnomalyDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    output = model(batch_x)
                    _, predicted = torch.max(output.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/100], Loss: {total_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), 'anomaly_detector.pth')
    print("✅ Anomaly detector saved as anomaly_detector.pth")
    return model

def train_classifier(X, y_multiclass):
    print("\n🚀 Training Anomaly Classifier...")
    
    # 只使用异常样本
    anomaly_indices = np.where(y_multiclass > 0)[0]
    X_anomaly = X[anomaly_indices]
    y_anomaly = y_multiclass[anomaly_indices] - 1  # 转换为0-5
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_anomaly, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly
    )
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 训练模型
    model = AnomalyClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    output = model(batch_x)
                    _, predicted = torch.max(output.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/100], Loss: {total_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), 'anomaly_classifier.pth')
    print("✅ Anomaly classifier saved as anomaly_classifier.pth")
    return model

def main():
    # 生成数据
    X, y_binary, y_multiclass = generate_data()
    
    # 训练检测器
    detector = train_detector(X, y_binary)
    
    # 训练分类器
    classifier = train_classifier(X, y_multiclass)
    
    print("\n🎉 Training complete!")
    print("📁 Generated files:")
    print("   - anomaly_detector.pth")
    print("   - anomaly_classifier.pth")
    print("   - separate_models_scaler.pkl")

if __name__ == "__main__":
    main() 