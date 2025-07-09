#!/bin/bash

# ================================================================================================
# 移动设备DLC推理程序编译脚本
# ================================================================================================

set -e  # 出错时退出

echo "=== DLC Mobile Inference Build Script ==="

# 获取脚本所在目录（项目根目录）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project Root: $PROJECT_ROOT"

# 设置SNPE SDK路径
SNPE_SDK_DIR="$PROJECT_ROOT/2.26.2.240911"

# 检查SNPE SDK是否存在
if [ ! -d "$SNPE_SDK_DIR" ]; then
    echo "Error: SNPE SDK directory not found at $SNPE_SDK_DIR"
    echo "Please ensure the SNPE SDK is extracted in the project root directory"
    exit 1
fi

echo "SNPE SDK Path: $SNPE_SDK_DIR"

# 自动检测系统架构
ARCH=$(uname -m)
OS=$(uname -s)

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"

# 选择合适的库路径
if [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        LIB_ARCH="x86_64-linux-clang"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        LIB_ARCH="aarch64-ubuntu-gcc9.4"
    else
        echo "Warning: Unsupported architecture $ARCH, using x86_64-linux-clang"
        LIB_ARCH="x86_64-linux-clang"
    fi
else
    echo "Warning: Unsupported OS $OS, using x86_64-linux-clang"
    LIB_ARCH="x86_64-linux-clang"
fi

echo "Using library architecture: $LIB_ARCH"

# 验证库目录是否存在
LIB_DIR="$SNPE_SDK_DIR/lib/$LIB_ARCH"
if [ ! -d "$LIB_DIR" ]; then
    echo "Error: Library directory not found at $LIB_DIR"
    echo "Available architectures:"
    ls -la "$SNPE_SDK_DIR/lib/"
    exit 1
fi

# 设置编译参数
CXX=g++
CXXFLAGS="-std=c++11 -O2 -fPIC -Wall -Wno-deprecated-declarations"

# 设置头文件路径 - 修正为实际的SNPE头文件路径
INCLUDE_PATHS=(
    "-I$SNPE_SDK_DIR/include/SNPE"
    "-I$SNPE_SDK_DIR/include"
)

# 设置库文件路径
LIB_PATHS=(
    "-L$LIB_DIR"
    "-Wl,-rpath,$LIB_DIR"
)

# 设置链接库
LIBS=(
    "-lSNPE"
    "-lstdc++"
    "-lm"
    "-lpthread"
    "-ldl"
)

# 源文件和目标文件
SOURCE_FILE="$PROJECT_ROOT/dlc_mobile_inference.cpp"
TARGET_FILE="$PROJECT_ROOT/dlc_mobile_inference"

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file $SOURCE_FILE not found"
    exit 1
fi

# 检查必要的头文件是否存在
REQUIRED_HEADERS=(
    "$SNPE_SDK_DIR/include/SNPE/SNPE/SNPE.hpp"
    "$SNPE_SDK_DIR/include/SNPE/SNPE/SNPEFactory.hpp"
    "$SNPE_SDK_DIR/include/SNPE/DlContainer/IDlContainer.hpp"
    "$SNPE_SDK_DIR/include/SNPE/DlSystem/TensorMap.hpp"
)

echo "Checking required header files..."
for header in "${REQUIRED_HEADERS[@]}"; do
    if [ ! -f "$header" ]; then
        echo "Error: Required header file not found: $header"
        exit 1
    fi
    echo "✓ Found: $(basename "$header")"
done

# 检查必要的库文件是否存在
REQUIRED_LIBS=(
    "$LIB_DIR/libSNPE.so"
)

echo "Checking required library files..."
for lib in "${REQUIRED_LIBS[@]}"; do
    if [ ! -f "$lib" ]; then
        echo "Error: Required library file not found: $lib"
        echo "Available libraries in $LIB_DIR:"
        ls -la "$LIB_DIR/"
        exit 1
    fi
    echo "✓ Found: $(basename "$lib")"
done

echo "Compiling $SOURCE_FILE..."
echo "Target Architecture: $LIB_ARCH"
echo "Output: $TARGET_FILE"

# 编译命令
echo ""
echo "Build command:"
echo "$CXX $CXXFLAGS ${INCLUDE_PATHS[*]} ${LIB_PATHS[*]} \"$SOURCE_FILE\" -o \"$TARGET_FILE\" ${LIBS[*]}"
echo ""

# 执行编译
$CXX $CXXFLAGS "${INCLUDE_PATHS[@]}" "${LIB_PATHS[@]}" "$SOURCE_FILE" -o "$TARGET_FILE" "${LIBS[@]}"

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "Executable: $TARGET_FILE"
    
    # 检查生成的可执行文件
    if [ -f "$TARGET_FILE" ]; then
        echo "File size: $(du -h "$TARGET_FILE" | cut -f1)"
        echo "Permissions: $(ls -la "$TARGET_FILE")"
        
        # 设置执行权限
        chmod +x "$TARGET_FILE"
        echo "Execute permissions set"
    fi
    
    # 检查DLC模型文件是否存在
    echo ""
    echo "📋 Checking DLC model files..."
    DLC_FILES=(
        "$PROJECT_ROOT/realistic_end_to_end_anomaly_detector.dlc"
        "$PROJECT_ROOT/realistic_end_to_end_anomaly_classifier.dlc"
    )
    
    for dlc_file in "${DLC_FILES[@]}"; do
        if [ -f "$dlc_file" ]; then
            echo "✓ Found: $(basename "$dlc_file") ($(du -h "$dlc_file" | cut -f1))"
        else
            echo "⚠️  Missing: $(basename "$dlc_file")"
        fi
    done
    
    echo ""
    echo "🚀 Usage:"
    echo "  cd \"$PROJECT_ROOT\""
    echo "  ./dlc_mobile_inference <detector.dlc> <classifier.dlc> <input_data.bin>"
    echo ""
    echo "📝 Example:"
    echo "  ./dlc_mobile_inference realistic_end_to_end_anomaly_detector.dlc realistic_end_to_end_anomaly_classifier.dlc input_sample.bin"
    echo ""
    echo "💡 Generate test data:"
    echo "  python3 generate_test_input.py"
    
else
    echo "❌ Build failed!"
    exit 1
fi

echo "=== Build Complete ===" 