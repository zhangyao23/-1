# 项目文档索引

本索引提供了项目中所有文档的快速导航，帮助您快速找到所需的信息。

## 🚀 快速导航

### 新手入门
- **[README.md](README.md)** - 项目主文档，包含概述和快速开始
- **[data/README.md](data/README.md)** - 数据格式说明和示例
- **[guide/模型集成指南.md](guide/模型集成指南.md)** - 模型集成步骤

### 技术开发
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - 项目技术总结和架构说明
- **[guide/model_architecture_guide.md](guide/model_architecture_guide.md)** - 模型架构详细说明
- **[guide/model_runtime_explanation.md](guide/model_runtime_explanation.md)** - 模型运行时技术细节

### 格式规范
- **[INPUT_FORMAT_SPECIFICATION.md](INPUT_FORMAT_SPECIFICATION.md)** - 输入数据格式规范
- **[OUTPUT_FORMAT_SPECIFICATION.md](OUTPUT_FORMAT_SPECIFICATION.md)** - 输出结果格式规范
- **[FORMAT_SPECIFICATIONS_INDEX.md](FORMAT_SPECIFICATIONS_INDEX.md)** - 格式规范总索引

### 部署和测试
- **[MOBILE_DEPLOYMENT_GUIDE.md](MOBILE_DEPLOYMENT_GUIDE.md)** - 移动端部署指南
- **[guide/cpp_verification_guide.md](guide/cpp_verification_guide.md)** - C++环境验证指南
- **[guide/手动测试指南.md](guide/手动测试指南.md)** - 手动测试步骤

### 项目汇报
- **[项目汇报内容大纲.md](项目汇报内容大纲.md)** - PPT制作大纲
- **[TASK_COMPLETION_SUMMARY.md](TASK_COMPLETION_SUMMARY.md)** - 任务完成总结
- **[VERSION_v2.0_FINAL.md](VERSION_v2.0_FINAL.md)** - 版本说明

## 📋 按功能分类

### 核心文档
| 文档 | 用途 | 适用人群 | 方案归属 |
|------|------|----------|----------|
| [README.md](README.md) | 项目主文档 | 所有用户 | 现用（多层神经网络） |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | 技术总结 | 开发人员 | 现用（多层神经网络） |
| [项目汇报内容大纲.md](项目汇报内容大纲.md) | 汇报材料 | 项目汇报 | 现用（多层神经网络） |

### 数据相关
| 文档 | 用途 | 适用人群 | 方案归属 |
|------|------|----------|----------|
| [data/README.md](data/README.md) | 数据格式说明 | 数据准备人员 | 通用 |
| [data/example_data.json](data/example_data.json) | 数据格式示例 | 数据准备人员 | 通用 |
| [INPUT_FORMAT_SPECIFICATION.md](INPUT_FORMAT_SPECIFICATION.md) | 输入格式规范 | 开发人员 | 通用 |

### 技术指南
| 文档 | 用途 | 适用人群 | 方案归属 |
|------|------|----------|----------|
| [guide/模型集成指南.md](guide/模型集成指南.md) | 模型集成 | 集成工程师 | 现用（多层神经网络） |
| [guide/model_architecture_guide.md](guide/model_architecture_guide.md) | 模型架构 | AI工程师 | 现用+历史（前部为现用，后部为历史） |
| [guide/end_to_end_model_workflow.md](guide/end_to_end_model_workflow.md) | 端到端开发流程 | 开发人员 | 现用（多层神经网络） |
| [guide/python_model_training_guide.md](guide/python_model_training_guide.md) | Python训练流程 | 开发人员 | 现用（多层神经网络） |
| [guide/model_runtime_explanation.md](guide/model_runtime_explanation.md) | 运行时细节 | AI工程师 | 现用（多层神经网络） |
| [guide/simulation_guide.md](guide/simulation_guide.md) | 仿真与测试 | 测试工程师 | 现用（多层神经网络） |
| [guide/手动测试指南.md](guide/手动测试指南.md) | 手动测试 | 测试工程师 | 现用（多层神经网络） |
| [guide/two_tier_detection_guide.md](guide/two_tier_detection_guide.md) | 两阶段架构 | AI工程师 | 历史（自编码器/随机森林） |
| [guide/cpp_files_functionality.md](guide/cpp_files_functionality.md) | C++功能说明 | C++开发人员 | 历史（自研C++推理/双模型） |
| [guide/cpp_verification_guide.md](guide/cpp_verification_guide.md) | C++验证 | 测试工程师 | 历史（自研C++推理/双模型） |
| [guide/quick_cpp_verification.md](guide/quick_cpp_verification.md) | C++快速验证 | 测试工程师 | 历史（自研C++推理/双模型） |

### 部署和测试
| 文档 | 用途 | 适用人群 | 方案归属 |
|------|------|----------|----------|
| [MOBILE_DEPLOYMENT_GUIDE.md](MOBILE_DEPLOYMENT_GUIDE.md) | 移动端部署 | 部署工程师 | 现用（多层神经网络） |
| [guide/cpp_verification_guide.md](guide/cpp_verification_guide.md) | C++验证 | 测试工程师 | 历史（自研C++推理/双模型） |
| [SNPE_VERIFICATION_REPORT.md](SNPE_VERIFICATION_REPORT.md) | SNPE验证报告 | 测试工程师 | 通用 |

### 项目管理
| 文档 | 用途 | 适用人群 | 方案归属 |
|------|------|----------|----------|
| [NEXT_TASKS_ROADMAP.md](NEXT_TASKS_ROADMAP.md) | 后续任务规划 | 项目经理 | 通用 |
| [TASK_COMPLETION_SUMMARY.md](TASK_COMPLETION_SUMMARY.md) | 任务完成总结 | 项目经理 | 通用 |
| [CORE_FILES_v2.0.txt](CORE_FILES_v2.0.txt) | 核心文件列表 | 开发人员 | 通用 |

> 注：
> - “现用（多层神经网络）”为当前主推、推荐使用的方案及相关文档。
> - “历史（自编码器/随机森林/双模型）”为早期探索、已被替代的方案，仅供参考。
> - “现用+历史”表示该文档同时包含现用和历史方案内容，具体请参见文档内章节说明。
> - “通用”表示适用于所有方案。

## 🎯 使用场景指南

### 场景1: 我是新手，想了解项目
**推荐阅读顺序**:
1. [README.md](README.md) - 了解项目概况
2. [data/README.md](data/README.md) - 了解数据格式
3. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 了解技术细节

### 场景2: 我要集成模型到现有系统
**推荐阅读顺序**:
1. [guide/模型集成指南.md](guide/模型集成指南.md) - 集成步骤
2. [INPUT_FORMAT_SPECIFICATION.md](INPUT_FORMAT_SPECIFICATION.md) - 输入格式
3. [OUTPUT_FORMAT_SPECIFICATION.md](OUTPUT_FORMAT_SPECIFICATION.md) - 输出格式

### 场景3: 我要准备项目汇报
**推荐阅读顺序**:
1. [项目汇报内容大纲.md](项目汇报内容大纲.md) - PPT大纲
2. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 技术要点
3. [TASK_COMPLETION_SUMMARY.md](TASK_COMPLETION_SUMMARY.md) - 项目成果

### 场景4: 我要进行技术开发
**推荐阅读顺序**:
1. [guide/model_architecture_guide.md](guide/model_architecture_guide.md) - 架构说明
2. [guide/model_runtime_explanation.md](guide/model_runtime_explanation.md) - 运行时细节
3. [guide/cpp_files_functionality.md](guide/cpp_files_functionality.md) - C++接口

### 场景5: 我要部署到移动设备
**推荐阅读顺序**:
1. [MOBILE_DEPLOYMENT_GUIDE.md](MOBILE_DEPLOYMENT_GUIDE.md) - 部署指南
2. [guide/cpp_verification_guide.md](guide/cpp_verification_guide.md) - 环境验证
3. [guide/手动测试指南.md](guide/手动测试指南.md) - 测试验证

## 📊 文档统计

- **总文档数**: 20+
- **核心文档**: 3个
- **技术指南**: 9个
- **格式规范**: 3个
- **部署文档**: 2个
- **管理文档**: 4个

## 🔍 搜索建议

### 按关键词搜索
- **数据格式**: 搜索 `data/README.md`, `INPUT_FORMAT_SPECIFICATION.md`
- **模型集成**: 搜索 `guide/模型集成指南.md`, `MOBILE_DEPLOYMENT_GUIDE.md`
- **技术架构**: 搜索 `PROJECT_SUMMARY.md`, `guide/model_architecture_guide.md`
- **测试验证**: 搜索 `guide/cpp_verification_guide.md`, `guide/手动测试指南.md`
- **项目汇报**: 搜索 `项目汇报内容大纲.md`, `TASK_COMPLETION_SUMMARY.md`

### 按文件类型搜索
- **.md**: 所有文档文件
- **guide/**: 技术指南文档
- **data/**: 数据相关文档
- **根目录**: 核心项目文档

---

*如有疑问，请参考主README.md或联系项目维护者。* 