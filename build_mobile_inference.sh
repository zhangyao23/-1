#!/bin/bash

# ================================================================================================
# 移动设备DLC推理程序编译脚本
# ================================================================================================

set -e  # 出错时退出

echo "=== DLC Mobile Inference Build Script ==="

# 检查SNPE SDK路径
if [ -z "$SNPE_ROOT" ]; then
    echo "Error: SNPE_ROOT environment variable not set"
    echo "Please set SNPE_ROOT to your SNPE SDK installation path"
    echo "Example: export SNPE_ROOT=/path/to/snpe-2.26.2.240911"
    exit 1
fi

echo "SNPE SDK Path: $SNPE_ROOT"

# 设置编译参数
CXX=g++
CXXFLAGS="-std=c++11 -O2 -fPIC -Wall"
TARGET_ARCH="x86_64-linux-clang"  # 可修改为arm64-android等

# 设置头文件路径
INCLUDE_PATHS=(
    "-I$SNPE_ROOT/include/zdl"
    "-I$SNPE_ROOT/include"
)

# 设置库文件路径
LIB_PATHS=(
    "-L$SNPE_ROOT/lib/$TARGET_ARCH"
    "-L$SNPE_ROOT/bin/$TARGET_ARCH"
)

# 设置链接库
LIBS=(
    "-lSNPE"
    "-lhta"
    "-lstdc++"
    "-lm"
    "-lpthread"
    "-ldl"
)

# 源文件和目标文件
SOURCE_FILE="dlc_mobile_inference.cpp"
TARGET_FILE="dlc_mobile_inference"

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file $SOURCE_FILE not found"
    exit 1
fi

echo "Compiling $SOURCE_FILE..."
echo "Target Architecture: $TARGET_ARCH"

# 编译命令
echo "Build command:"
echo "$CXX $CXXFLAGS ${INCLUDE_PATHS[@]} ${LIB_PATHS[@]} $SOURCE_FILE -o $TARGET_FILE ${LIBS[@]}"
echo ""

# 执行编译
$CXX $CXXFLAGS "${INCLUDE_PATHS[@]}" "${LIB_PATHS[@]}" "$SOURCE_FILE" -o "$TARGET_FILE" "${LIBS[@]}"

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "Executable: $TARGET_FILE"
    
    # 检查生成的可执行文件
    if [ -f "$TARGET_FILE" ]; then
        echo "File size: $(du -h $TARGET_FILE | cut -f1)"
        echo "Permissions: $(ls -la $TARGET_FILE)"
        
        # 设置执行权限
        chmod +x "$TARGET_FILE"
        echo "Execute permissions set"
    fi
    
    echo ""
    echo "🚀 Usage:"
    echo "  ./$TARGET_FILE <detector.dlc> <classifier.dlc> <input_data.bin>"
    echo ""
    echo "📝 Example:"
    echo "  ./$TARGET_FILE realistic_end_to_end_anomaly_detector.dlc realistic_end_to_end_anomaly_classifier.dlc input_sample.bin"
    
else
    echo "❌ Build failed!"
    exit 1
fi

echo "=== Build Complete ===" 