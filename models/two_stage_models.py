import torch
import torch.nn as nn

class AnomalyDetectionNetwork(nn.Module):
    """
    第一阶段：异常检测网络 (2分类)
    输入: 6维网络特征
    输出: 2个类别 (normal, anomaly)
    """
    def __init__(self, input_dim=6):
        super(AnomalyDetectionNetwork, self).__init__()
        
        # 专门用于异常检测的网络架构
        self.detector = nn.Sequential(
            nn.Linear(input_dim, 64),
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
            
            nn.Linear(16, 2)  # 2分类: normal vs anomaly
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
        return self.detector(x)

class AnomalyClassificationNetwork(nn.Module):
    """
    第二阶段：异常分类网络 (6分类)
    输入: 6维网络特征
    输出: 6个异常类别
    """
    def __init__(self, input_dim=6, n_classes=6):
        super(AnomalyClassificationNetwork, self).__init__()
        
        # 专门用于异常分类的网络架构
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
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
            
            nn.Linear(32, n_classes)  # 6分类: 6种异常类型
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
        return self.classifier(x)

class TwoStageAnomalySystem:
    """
    两阶段异常检测系统
    """
    def __init__(self, detector_model, classifier_model, scaler=None):
        self.detector = detector_model
        self.classifier = classifier_model
        self.scaler = scaler
        
        # 类别标签
        self.binary_labels = ['normal', 'anomaly']
        self.anomaly_labels = ['connection_timeout', 'mixed_anomaly', 'network_congestion', 
                             'packet_corruption', 'resource_overload', 'signal_degradation']
    
    def predict(self, features):
        """
        两阶段预测
        """
        # 标准化特征
        if self.scaler:
            features = self.scaler.transform(features)
        
        # 阶段1: 异常检测
        with torch.no_grad():
            detector_input = torch.FloatTensor(features)
            detector_output = self.detector(detector_input)
            detector_probs = torch.softmax(detector_output, dim=1)
            
            # 判断是否异常
            is_anomaly = torch.argmax(detector_probs, dim=1).item()
            detection_confidence = detector_probs[0][is_anomaly].item()
            
            if is_anomaly == 0:  # normal
                return {
                    'status': 'normal',
                    'confidence': detection_confidence,
                    'anomaly_type': None,
                    'anomaly_confidence': None
                }
            else:  # anomaly
                # 阶段2: 异常分类
                classifier_input = torch.FloatTensor(features)
                classifier_output = self.classifier(classifier_input)
                classifier_probs = torch.softmax(classifier_output, dim=1)
                
                anomaly_type_idx = torch.argmax(classifier_probs, dim=1).item()
                anomaly_confidence = classifier_probs[0][anomaly_type_idx].item()
                
                return {
                    'status': 'anomaly',
                    'confidence': detection_confidence,
                    'anomaly_type': self.anomaly_labels[anomaly_type_idx],
                    'anomaly_confidence': anomaly_confidence
                } 