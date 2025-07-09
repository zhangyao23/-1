#!/bin/bash

# ================================================================================================
# 项目快速设置脚本
# ================================================================================================

set -e

echo "🚀 AI网络异常检测系统 - 项目设置"
echo "=" * 50

# 获取脚本所在目录（项目根目录）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📍 项目根目录: $PROJECT_ROOT"

# 创建必要的目录结构
echo "📁 创建项目目录结构..."
REQUIRED_DIRS=(
    "models"
    "data"
    "test"
    "guide"
    "config"
    "src"
    "logs"
    "scripts"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    mkdir -p "$PROJECT_ROOT/$dir"
    echo "✅ 创建目录: $dir/"
done

# 设置脚本权限
echo "⚙️  设置脚本权限..."
SCRIPTS=(
    "build_mobile_inference.sh"
    "setup_project.sh"
    "scripts/verify_project_paths.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$PROJECT_ROOT/$script" ]; then
        chmod +x "$PROJECT_ROOT/$script"
        echo "✅ 设置可执行权限: $script"
    fi
done

# 检查SNPE SDK
echo "🔧 检查SNPE SDK..."
SNPE_SDK_DIR="$PROJECT_ROOT/2.26.2.240911"

if [ ! -d "$SNPE_SDK_DIR" ]; then
    echo "⚠️  SNPE SDK目录不存在: $SNPE_SDK_DIR"
    echo "请确保SNPE SDK已解压到项目根目录"
    echo "预期目录结构: $PROJECT_ROOT/2.26.2.240911/"
else
    echo "✅ SNPE SDK已就绪"
fi

# 检查Python环境
echo "🐍 检查Python环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Python环境: $PYTHON_VERSION"
    
    # 检查必要的Python包
    echo "📦 检查Python依赖..."
    REQUIRED_PACKAGES=(
        "numpy"
        "tensorflow"
        "scikit-learn"
        "joblib"
        "pandas"
    )
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python3 -c "import $package" &> /dev/null; then
            echo "✅ Python包: $package"
        else
            echo "⚠️  Python包缺失: $package"
            echo "   安装命令: pip3 install $package"
        fi
    done
else
    echo "❌ Python3未安装"
    echo "请安装Python 3.8或更高版本"
fi

# 检查编译环境
echo "⚙️  检查编译环境..."
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    echo "✅ 编译器: $GCC_VERSION"
else
    echo "❌ g++编译器未安装"
    echo "Ubuntu/Debian: sudo apt-get install build-essential"
    echo "CentOS/RHEL: sudo yum install gcc-c++"
fi

# 检查关键文件
echo "📋 检查关键文件..."
CRITICAL_FILES=(
    "dlc_mobile_inference.cpp"
    "build_mobile_inference.sh"
    "realistic_end_to_end_anomaly_detector.dlc"
    "realistic_end_to_end_anomaly_classifier.dlc"
    "realistic_raw_data_scaler.pkl"
    "README.md"
    "generate_test_input.py"
)

MISSING_FILES=()
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        echo "✅ 关键文件: $file"
    else
        echo "❌ 缺失文件: $file"
        MISSING_FILES+=("$file")
    fi
done

# 运行项目验证
echo "🔍 运行项目验证..."
if [ -f "$PROJECT_ROOT/scripts/verify_project_paths.py" ]; then
    python3 "$PROJECT_ROOT/scripts/verify_project_paths.py"
else
    echo "⚠️  验证脚本不存在，跳过详细验证"
fi

# 生成快速测试脚本
echo "📝 生成快速测试脚本..."
cat > "$PROJECT_ROOT/quick_test.sh" << 'EOF'
#!/bin/bash
# 快速测试脚本

echo "🚀 快速测试AI网络异常检测系统"
echo "=" * 40

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "1. 运行项目验证..."
if [ -f "scripts/verify_project_paths.py" ]; then
    python3 scripts/verify_project_paths.py
fi

echo "2. 生成测试数据..."
if [ -f "generate_test_input.py" ]; then
    python3 generate_test_input.py
fi

echo "3. 编译C++推理程序..."
if [ -f "build_mobile_inference.sh" ]; then
    ./build_mobile_inference.sh
fi

echo "4. 测试端到端系统..."
if [ -f "test_realistic_end_to_end_system.py" ]; then
    python3 test_realistic_end_to_end_system.py
fi

echo "✅ 快速测试完成"
EOF

chmod +x "$PROJECT_ROOT/quick_test.sh"
echo "✅ 快速测试脚本已创建: quick_test.sh"

# 输出总结
echo ""
echo "🎯 设置总结"
echo "=" * 50

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo "✅ 项目设置完成！所有关键文件都已就绪"
    echo ""
    echo "🚀 接下来的步骤:"
    echo "1. 运行快速测试: ./quick_test.sh"
    echo "2. 编译推理程序: ./build_mobile_inference.sh"
    echo "3. 生成测试数据: python3 generate_test_input.py"
    echo "4. 运行推理测试: ./dlc_mobile_inference detector.dlc classifier.dlc input.bin"
    echo ""
    echo "📚 更多信息请查看:"
    echo "- README.md - 项目完整说明"
    echo "- MOBILE_DEPLOYMENT_GUIDE.md - 移动设备部署指南"
    echo "- guide/cpp_files_functionality.md - C++代码功能说明"
else
    echo "⚠️  项目设置部分完成，但有些文件缺失:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "请确保所有必要文件都已就绪后再继续"
fi

echo ""
echo "🔗 有用的命令:"
echo "- 项目验证: python3 scripts/verify_project_paths.py"
echo "- 快速测试: ./quick_test.sh"
echo "- 编译程序: ./build_mobile_inference.sh"
echo "- C++验证: python3 test/quick_cpp_test.py"

echo ""
echo "📞 如遇问题，请检查:"
echo "1. SNPE SDK是否正确解压到项目根目录"
echo "2. Python依赖是否已安装"
echo "3. 编译环境是否配置正确"
echo "4. DLC模型文件是否存在" 