# C++快速验证指南

## 快速验证你的C++代码

### 🚀 一键快速验证

```bash
python3 test/quick_cpp_test.py
```

### ✅ 成功输出示例

```
🚀 快速C++功能验证
1. 检查文件...
  ✅ dlc_mobile_inference.cpp (20985 bytes)
  ✅ build_mobile_inference.sh (2402 bytes)
  ✅ realistic_end_to_end_anomaly_detector.dlc (58482 bytes)
  ✅ realistic_end_to_end_anomaly_classifier.dlc (194754 bytes)

2. C++结构检查...
  ✅ 所有关键组件都存在

3. C++语法检查...
  ✅ 基本语法正确（缺少SNPE头文件，这是正常的）
  ℹ️  完整编译需要SNPE SDK环境

4. 生成测试数据...
  ✅ 测试数据生成: test_input_quick.bin

5. 检查编译脚本...
  ✅ 编译脚本存在
  ⚠️  SNPE_ROOT未设置，跳过编译测试

6. 清理临时文件

🎉 快速验证完成！
✅ 快速验证通过 - C++代码基本功能正常
```

### 📋 验证内容

- **文件检查**: 确保所有必要文件存在
- **结构验证**: 检查C++代码包含所有关键组件
- **语法检查**: 验证代码基本语法正确
- **测试数据**: 生成和清理测试数据
- **编译脚本**: 检查编译配置

### 🔧 如果有SNPE环境

```bash
# 设置SNPE环境
export SNPE_ROOT=/path/to/snpe-2.26.2.240911

# 再次运行验证（会进行实际编译和推理测试）
python3 test/quick_cpp_test.py
```

### 🚨 常见问题

**Q: 语法检查失败怎么办？**
A: 检查C++代码是否有语法错误，确保使用了正确的头文件包含。

**Q: 文件缺失怎么办？**
A: 确保所有必要文件都在项目根目录中，重新下载缺失文件。

**Q: 需要完整测试怎么办？**
A: 运行完整验证脚本：`python3 test/verify_cpp_functionality.py`

### 📖 更多信息

详细验证指南：[guide/cpp_verification_guide.md](cpp_verification_guide.md) 