import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib

# --- 1. 定义多任务模型 ---
class MultiTaskAnomalyModel(nn.Module):
    def __init__(self):
        super(MultiTaskAnomalyModel, self).__init__()
        
        # 共享主干网络
        self.shared_layers = nn.Sequential(
            nn.Linear(11, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 输出头1: 异常检测 (2个输出: 异常, 正常)
        self.detection_head = nn.Linear(64, 2)
        
        # 输出头2: 异常分类 (6个输出: 6种异常类型)
        self.classification_head = nn.Linear(64, 6)

    def forward(self, x):
        # 数据通过共享层
        features = self.shared_layers(x)
        
        # 特征分别进入两个输出头
        detection_output = self.detection_head(features)
        classification_output = self.classification_head(features)
        
        # 将两个输出合并成一个张量
        combined_output = torch.cat((detection_output, classification_output), dim=1)
        
        return combined_output

# --- 2. 数据准备 ---
def generate_and_prepare_data():
    print("🔄 Generating and preparing data...")
    # 使用与之前相同的真实数据生成逻辑
    from train_realistic_end_to_end_networks import generate_realistic_network_data
    
    X, y_binary, y_multiclass = generate_realistic_network_data(n_samples=30000)
    
    # 标准化11维原始数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'multitask_scaler.pkl')
    print("✅ Scaler saved to multitask_scaler.pkl")

    # 分割数据集
    X_train, X_test, y_binary_train, y_binary_test, y_multiclass_train, y_multiclass_test = train_test_split(
        X_scaled, y_binary, y_multiclass, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_det_train = torch.LongTensor(y_binary_train)
    y_det_test = torch.LongTensor(y_binary_test)
    y_cls_train = torch.LongTensor(y_multiclass_train)
    y_cls_test = torch.LongTensor(y_multiclass_test)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_det_train, y_cls_train)
    test_dataset = TensorDataset(X_test_tensor, y_det_test, y_cls_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

# --- 3. 训练逻辑 ---
def train_model(model, train_loader, test_loader, epochs=120, lr=0.0005): # 降低学习率，增加epoch
    print("🚀 Starting model training (with data augmentation and stronger regularization)...")
    
    # 定义损失函数
    detection_criterion = nn.CrossEntropyLoss()
    # 对于分类任务，我们只关心异常样本的损失
    # `ignore_index=0` 会让损失函数忽略所有标签为0（即“正常”）的样本
    classification_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # 增加权重衰减
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_det_y, batch_cls_y in train_loader:
            
            # --- 数据增强：在训练时添加少量噪声 ---
            noise = torch.randn_like(batch_x) * 0.05 # 5%的噪声
            batch_x_augmented = batch_x + noise

            optimizer.zero_grad()
            
            # 获取模型输出
            combined_output = model(batch_x_augmented)
            
            # 从合并的输出中分离出两个任务的结果
            detection_output = combined_output[:, :2]
            classification_output = combined_output[:, 2:]

            # 计算两个任务的损失
            loss_det = detection_criterion(detection_output, batch_det_y)

            # 计算分类任务的损失 (只对异常样本)
            anomaly_mask = (batch_det_y == 1)
            if anomaly_mask.any():
                # 注意：分类头的目标是0-5，所以原始标签（1-6）需要减1
                loss_cls = classification_criterion(
                    classification_output[anomaly_mask], 
                    batch_cls_y[anomaly_mask] - 1
                )
            else:
                # 如果这个批次里没有异常样本，则分类损失为0
                loss_cls = torch.tensor(0.0, device=batch_x.device)

            # 将两个损失加权相加
            # 在这个场景中，检测任务更基础，所以给它稍高的权重
            loss = 0.6 * loss_det + 0.4 * loss_cls
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # 在每个epoch后进行评估
        if (epoch + 1) % 10 == 0:
            det_acc, cls_acc = evaluate_model(model, test_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, "
                  f"Detection Acc: {det_acc:.2f}%, Classification Acc: {cls_acc:.2f}%")

# --- 4. 评估逻辑 ---
def evaluate_model(model, test_loader):
    model.eval()
    
    correct_det = 0
    total_det = 0
    correct_cls = 0
    total_cls = 0
    
    with torch.no_grad():
        for batch_x, batch_det_y, batch_cls_y in test_loader:
            combined_output = model(batch_x)
            
            # 从合并的输出中分离
            detection_output = combined_output[:, :2]
            classification_output = combined_output[:, 2:]

            # 评估检测任务
            _, predicted_det = torch.max(detection_output.data, 1)
            total_det += batch_det_y.size(0)
            correct_det += (predicted_det == batch_det_y).sum().item()
            
            # 评估分类任务 (只在异常样本上)
            anomaly_mask = (batch_det_y == 1)
            if anomaly_mask.any():
                _, predicted_cls = torch.max(classification_output[anomaly_mask].data, 1)
                # 分类标签需要减1
                true_cls_labels = batch_cls_y[anomaly_mask] - 1
                total_cls += true_cls_labels.size(0)
                correct_cls += (predicted_cls == true_cls_labels).sum().item()

    det_accuracy = 100 * correct_det / total_det if total_det > 0 else 0
    cls_accuracy = 100 * correct_cls / total_cls if total_cls > 0 else 0
    
    return det_accuracy, cls_accuracy

# --- 5. 主函数 ---
def main():
    # 准备数据
    train_loader, test_loader = generate_and_prepare_data()
    
    # 初始化模型
    model = MultiTaskAnomalyModel()
    
    # 训练模型
    train_model(model, train_loader, test_loader)
    
    # 保存模型为ONNX格式
    print("\n💾 Saving model to ONNX format...")
    
    # 需要一个虚拟输入来导出
    dummy_input = torch.randn(1, 11)
    
    # 指定输入和输出节点的名称
    input_names = ["input"]
    output_names = ["combined_output"] # 现在只有一个输出
    
    torch.onnx.export(model, 
                      (dummy_input,),
                      "multitask_model.onnx",
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11,
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'combined_output': {0: 'batch_size'}})
    
    print("✅ Model saved as multitask_model.onnx")
    print("\n🎉 Training complete!")
    print("Next steps:")
    print("1. Use `snpe-onnx-to-dlc` to convert multitask_model.onnx to a single DLC file.")
    print("2. Update the C++ application to use the new single DLC model.")

if __name__ == "__main__":
    main() 