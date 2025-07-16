#!/bin/bash

# 简化的DLC转换脚本
echo "🔄 转换异常检测模型为DLC格式..."
snpe-onnx-to-dlc -i anomaly_detector.onnx -o anomaly_detector.dlc

echo "🔄 转换异常分类模型为DLC格式..."
snpe-onnx-to-dlc -i anomaly_classifier.onnx -o anomaly_classifier.dlc

echo "✅ DLC转换完成！"
ls -la anomaly_*.dlc 