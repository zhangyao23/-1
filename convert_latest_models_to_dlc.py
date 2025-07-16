import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import subprocess

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


def convert_to_onnx(model, input_shape, output_path):
    """将PyTorch模型转换为ONNX格式"""
    print(f"转换模型为ONNX格式: {output_path}")
    
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✅ ONNX模型已保存: {output_path}")

def convert_to_dlc(onnx_path, dlc_path):
    """将ONNX模型转换为DLC格式"""
    print(f"转换ONNX为DLC格式: {dlc_path}")
    
    cmd = [
        './2.26.2.240911/bin/x86_64-linux-clang/snpe-onnx-to-dlc',
        '--input_network', onnx_path,
        '--output_path', dlc_path
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(f"✅ DLC模型已生成: {dlc_path}")
        
        if os.path.exists(dlc_path):
            file_size = os.path.getsize(dlc_path)
            print(f"   文件大小: {file_size / 1024:.1f} KB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ DLC转换失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ 错误: 'snpe-onnx-to-dlc' not found.")
        return False

def main():
    print("=== 将最新的分离模型转换为DLC格式 ===")
    
    print("🔄 加载最新的分离模型...")
    detector_model = AnomalyDetector()
    detector_model.load_state_dict(torch.load('anomaly_detector.pth', map_location='cpu'))
    detector_model.eval()
    
    classifier_model = AnomalyClassifier()
    classifier_model.load_state_dict(torch.load('anomaly_classifier.pth', map_location='cpu'))
    classifier_model.eval()
    print("✅ 模型加载完成")
    
    print("\n🔄 转换为ONNX格式...")
    convert_to_onnx(detector_model, (11,), 'anomaly_detector.onnx')
    convert_to_onnx(classifier_model, (11,), 'anomaly_classifier.onnx')
    
    print("\n🔄 转换为DLC格式...")
    detector_success = convert_to_dlc('anomaly_detector.onnx', 'anomaly_detector.dlc')
    classifier_success = convert_to_dlc('anomaly_classifier.onnx', 'anomaly_classifier.dlc')
    
    print("\n=== 转换结果汇总 ===")
    if detector_success and classifier_success:
        print("✅ 所有模型转换成功!")
        print("📁 生成的DLC文件:")
        print("   - anomaly_detector.dlc    (异常检测)")
        print("   - anomaly_classifier.dlc  (异常分类)")
        
        total_size = 0
        for filename in ['anomaly_detector.dlc', 'anomaly_classifier.dlc']:
            if os.path.exists(filename):
                total_size += os.path.getsize(filename)
        
        print(f"   📦 总DLC文件大小: {total_size / 1024:.1f} KB")
    else:
        print("❌ 部分模型转换失败")

if __name__ == "__main__":
    main()
