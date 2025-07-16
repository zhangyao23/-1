#!/bin/bash
# DLC转换脚本
# 需要安装SNPE SDK并设置环境变量

echo "🔄 转换异常检测模型为DLC格式..."

# 转换异常检测模型
snpe-onnx-to-dlc     -i anomaly_detector.onnx     -o anomaly_detector.dlc

echo "🔄 转换异常分类模型为DLC格式..."

# 转换异常分类模型  
snpe-onnx-to-dlc     -i anomaly_classifier.onnx     -o anomaly_classifier.dlc

echo "✅ DLC转换完成！"
echo "📁 生成的文件："
echo "   - anomaly_detector.dlc"
echo "   - anomaly_classifier.dlc"
