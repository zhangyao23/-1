import torch
import torch.nn as nn

# 定义模型类
class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.layers(x)

class AnomalyClassifier(nn.Module):
    def __init__(self):
        super(AnomalyClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 6)
        )
    def forward(self, x):
        return self.layers(x)

# 转换函数
def convert_to_onnx(model, model_path, onnx_path):
    print(f"🔄 Loading {model_path} and converting to {onnx_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        dummy_input = torch.randn(1, 11)
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"✅ Saved {onnx_path}")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}")
    except Exception as e:
        print(f"❌ An error occurred during conversion for {model_path}: {e}")

# 执行转换
convert_to_onnx(AnomalyDetector(), 'anomaly_detector.pth', 'anomaly_detector.onnx')
print("-" * 20)
convert_to_onnx(AnomalyClassifier(), 'anomaly_classifier.pth', 'anomaly_classifier.onnx')
