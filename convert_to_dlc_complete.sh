#!/bin/bash

# DLC转换脚本 - 完整版
# 用于将分别训练的异常检测和分类模型转换为DLC格式

set -e  # 遇到错误时退出

echo "🚀 开始DLC转换流程..."
echo "=================================================="

# 检查必要的文件是否存在
echo "📋 检查必要文件..."

required_files=(
    "anomaly_detector.onnx"
    "anomaly_classifier.onnx"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 错误：找不到文件 $file"
        echo "请先运行 python3 convert_separate_models_to_dlc.py 生成ONNX文件"
        exit 1
    else
        echo "✅ 找到文件: $file"
    fi
done

# 检查SNPE工具是否可用
echo ""
echo "🔧 检查SNPE工具..."

if ! command -v snpe-onnx-to-dlc &> /dev/null; then
    echo "❌ 错误：找不到 snpe-onnx-to-dlc 命令"
    echo ""
    echo "请确保："
    echo "1. 已安装SNPE SDK"
    echo "2. 已设置SNPE环境变量"
    echo "3. 已添加到PATH中"
    echo ""
    echo "典型的SNPE环境设置："
    echo "export SNPE_ROOT=/path/to/snpe"
    echo "export PATH=\$SNPE_ROOT/bin/x86_64-linux-clang:\$PATH"
    echo "export LD_LIBRARY_PATH=\$SNPE_ROOT/lib/x86_64-linux-clang:\$LD_LIBRARY_PATH"
    exit 1
else
    echo "✅ SNPE工具可用"
fi

# 显示SNPE版本信息
echo "📊 SNPE版本信息："
snpe-onnx-to-dlc --version 2>/dev/null || echo "无法获取版本信息"

echo ""
echo "🔄 开始转换模型..."

# 转换异常检测模型
echo ""
echo "📊 转换异常检测模型..."
echo "输入: anomaly_detector.onnx"
echo "输出: anomaly_detector.dlc"

if snpe-onnx-to-dlc -i anomaly_detector.onnx -o anomaly_detector.dlc; then
    echo "✅ 异常检测模型转换成功"
    
    # 显示文件信息
    if [ -f "anomaly_detector.dlc" ]; then
        file_size=$(du -h anomaly_detector.dlc | cut -f1)
        echo "   文件大小: $file_size"
    fi
else
    echo "❌ 异常检测模型转换失败"
    exit 1
fi

# 转换异常分类模型
echo ""
echo "📊 转换异常分类模型..."
echo "输入: anomaly_classifier.onnx"
echo "输出: anomaly_classifier.dlc"

if snpe-onnx-to-dlc -i anomaly_classifier.onnx -o anomaly_classifier.dlc; then
    echo "✅ 异常分类模型转换成功"
    
    # 显示文件信息
    if [ -f "anomaly_classifier.dlc" ]; then
        file_size=$(du -h anomaly_classifier.dlc | cut -f1)
        echo "   文件大小: $file_size"
    fi
else
    echo "❌ 异常分类模型转换失败"
    exit 1
fi

echo ""
echo "🎉 DLC转换完成！"
echo "=================================================="
echo "📁 生成的文件："
echo "   - anomaly_detector.dlc     (异常检测模型)"
echo "   - anomaly_classifier.dlc   (异常分类模型)"
echo ""
echo "📋 文件信息："
ls -lh anomaly_*.dlc 2>/dev/null || echo "无法获取文件信息"
echo ""
echo "💡 下一步："
echo "1. 将这两个.dlc文件复制到目标设备"
echo "2. 在C++代码中加载这两个模型"
echo "3. 先调用检测模型，如果检测到异常再调用分类模型"
echo ""
echo "📚 相关文件："
echo "   - separate_models_scaler.pkl  (数据标准化器)"
echo "   - test_separate_models.py     (测试脚本)"
echo ""
echo "✅ 转换流程完成！" 