# C++功能验证指南

## 概述

本指南介绍如何验证 `dlc_mobile_inference.cpp` 的完整功能。我们提供了两个验证脚本：

1. **快速验证** (`test/quick_cpp_test.py`) - 基本功能检查
2. **完整验证** (`test/verify_cpp_functionality.py`) - 全面功能测试

## 验证脚本说明

### 快速验证脚本

用途：快速检查基本功能，适合开发过程中的日常检查。

**运行方式：**
```bash
python test/quick_cpp_test.py
```

**验证内容：**
- 文件存在性检查
- C++语法验证
- 测试数据生成
- 编译脚本检查
- 基本推理测试（如果SNPE环境可用）

### 完整验证脚本

用途：全面的功能验证，适合正式测试和部署前检查。

**运行方式：**
```bash
python test/verify_cpp_functionality.py
```

**验证内容：**
- 文件存在性检查
- 编译测试（实际编译或语法检查）
- 多场景测试数据生成
- 推理执行测试
- 输出格式验证
- 性能测试
- 内存泄漏检查

## 环境要求

### 基本验证需求

- Python 3.6+
- g++ 编译器
- 必要的项目文件：
  - `dlc_mobile_inference.cpp`
  - `build_mobile_inference.sh`
  - `realistic_end_to_end_anomaly_detector.dlc`
  - `realistic_end_to_end_anomaly_classifier.dlc`

### 完整验证需求

**如果需要实际编译和运行：**
- SNPE SDK 2.26.2.240911 或更高版本
- 设置 `SNPE_ROOT` 环境变量
- 目标平台的交叉编译工具链

**可选工具：**
- `valgrind` - 用于内存泄漏检查
- `time` - 用于性能测试

## 使用步骤

### 1. 设置环境

```bash
# 设置SNPE环境（如果有）
export SNPE_ROOT=/path/to/snpe-2.26.2.240911

# 确保在项目根目录
cd /path/to/your/project
```

### 2. 运行快速验证

```bash
python test/quick_cpp_test.py
```

**预期输出：**
```
🚀 快速C++功能验证
1. 检查文件...
  ✅ dlc_mobile_inference.cpp
  ✅ build_mobile_inference.sh
  ✅ realistic_end_to_end_anomaly_detector.dlc
  ✅ realistic_end_to_end_anomaly_classifier.dlc

2. C++语法检查...
  ✅ 语法正确

3. 生成测试数据...
  ✅ 测试数据生成: test_input_quick.bin

4. 检查编译脚本...
  ✅ 编译脚本存在
  ✅ 编译成功
  ✅ 推理执行成功

5. 清理临时文件

🎉 快速验证完成！
✅ 快速验证通过 - C++代码基本功能正常
```

### 3. 运行完整验证

```bash
python test/verify_cpp_functionality.py
```

**预期输出：**
```
🚀 开始C++功能验证...
项目根目录: /path/to/project
结果目录: /path/to/project/test/cpp_verification_results

=== 步骤1: 检查文件存在性 ===
✅ PASS file_existence
    所有必要文件存在

=== 步骤2: 编译测试 ===
✅ PASS compilation
    编译成功，可执行文件大小: 1234567 bytes

=== 步骤3: 生成测试数据 ===
✅ PASS test_data_generation
    生成了5个测试场景

=== 步骤4: 推理执行测试 ===
✅ PASS inference_execution
    成功率: 5/5 (100.0%)

=== 步骤5: 输出格式验证 ===
✅ PASS output_validation
    有效输出: 5/5 (100.0%)

=== 步骤6: 性能测试 ===
✅ PASS performance_test
    平均执行时间: 1.00次/5次测试

=== 步骤7: 内存泄漏检查 ===
✅ PASS memory_leak_check
    无内存泄漏

=== 验证报告 ===
总测试数: 8
通过测试: 8
失败测试: 0
成功率: 100.0%

🎉 整体验证成功！C++功能正常
```

## 结果分析

### 测试结果目录结构

```
test/cpp_verification_results/
├── test_input_normal_network.bin        # 测试输入文件
├── test_input_normal_network.json       # 测试输入描述
├── test_input_wifi_degradation.bin      # 各种场景测试数据
├── test_input_wifi_degradation.json
├── ...
├── output_normal_network.txt             # 推理输出结果
├── output_wifi_degradation.txt
├── ...
└── verification_report.json             # 完整验证报告
```

### 验证报告格式

```json
{
  "timestamp": "2024-01-15 10:30:45",
  "project_root": "/path/to/project",
  "verification_results": {
    "file_existence": true,
    "compilation": true,
    "test_data_generation": true,
    "inference_execution": true,
    "output_validation": true,
    "performance_test": true,
    "memory_leak_check": true,
    "overall_success": true
  },
  "summary": {
    "total_tests": 8,
    "passed_tests": 8,
    "failed_tests": 0,
    "success_rate": 1.0
  }
}
```

## 常见问题排查

### 1. 编译失败

**症状：** `compilation` 测试失败

**可能原因：**
- SNPE SDK 未正确安装
- `SNPE_ROOT` 环境变量未设置
- 缺少必要的编译工具链

**解决方案：**
```bash
# 检查SNPE环境
echo $SNPE_ROOT
ls -la $SNPE_ROOT/include/zdl/

# 检查编译工具
g++ --version
```

### 2. 推理执行失败

**症状：** `inference_execution` 测试失败

**可能原因：**
- DLC模型文件损坏
- 输入数据格式不正确
- 运行时依赖缺失

**解决方案：**
```bash
# 检查模型文件
ls -la *.dlc
file realistic_end_to_end_anomaly_detector.dlc

# 检查输入数据
hexdump -C test_input_quick.bin
```

### 3. 输出格式验证失败

**症状：** `output_validation` 测试失败

**可能原因：**
- 输出处理逻辑错误
- JSON格式不正确
- 缺少必要的输出字段

**解决方案：**
```bash
# 检查输出文件
cat test/cpp_verification_results/output_normal_network.txt

# 验证JSON格式
python -m json.tool < output_file.json
```

### 4. 内存泄漏检查

**症状：** `memory_leak_check` 测试失败

**可能原因：**
- 内存管理错误
- 资源未正确释放

**解决方案：**
```bash
# 安装valgrind
sudo apt-get install valgrind

# 手动运行内存检查
valgrind --leak-check=full ./dlc_mobile_inference detector.dlc classifier.dlc input.bin
```

## 手动验证步骤

### 1. 手动编译

```bash
# 设置环境
export SNPE_ROOT=/path/to/snpe-2.26.2.240911

# 编译
chmod +x build_mobile_inference.sh
./build_mobile_inference.sh
```

### 2. 手动生成测试数据

```bash
# 使用Python生成测试数据
python -c "
import struct
data = [0.8, 0.75, 0.9, 100.0, 50.0, 200.0, 150.0, 20.0, 15.0, 0.3, 0.2]
with open('manual_test.bin', 'wb') as f:
    for val in data:
        f.write(struct.pack('<f', val))
"
```

### 3. 手动运行推理

```bash
# 运行推理
./dlc_mobile_inference \
    realistic_end_to_end_anomaly_detector.dlc \
    realistic_end_to_end_anomaly_classifier.dlc \
    manual_test.bin
```

### 4. 验证输出

```bash
# 检查输出是否包含期望的字段
grep -E "(detection_stage|classification_stage|confidence)" output.txt
```

## 自动化集成

### CI/CD 集成

```yaml
# .github/workflows/cpp-validation.yml
name: C++ Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y g++ valgrind
      - name: Quick validation
        run: python test/quick_cpp_test.py
      - name: Full validation
        run: python test/verify_cpp_functionality.py
        continue-on-error: true
```

### 预提交钩子

```bash
# .git/hooks/pre-commit
#!/bin/bash
python test/quick_cpp_test.py
if [ $? -ne 0 ]; then
    echo "❌ C++验证失败，提交被拒绝"
    exit 1
fi
```

## 最佳实践

1. **开发阶段：** 使用快速验证进行日常检查
2. **测试阶段：** 使用完整验证进行全面测试
3. **部署前：** 运行完整验证并检查所有输出
4. **持续集成：** 在CI/CD中集成验证脚本
5. **问题排查：** 保存验证结果用于问题分析

## 版本兼容性

- **Python：** 3.6+
- **SNPE SDK：** 2.26.2.240911+
- **g++：** 7.0+
- **操作系统：** Linux (Ubuntu 18.04+)

## 联系支持

如果在验证过程中遇到问题，请提供：
1. 验证脚本的完整输出
2. 系统环境信息
3. 错误日志文件
4. 验证报告JSON文件

通过这些信息，我们可以更好地帮助您解决问题。 