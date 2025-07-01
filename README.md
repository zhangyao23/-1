# AI 网络异常检测系统

本项目是一个基于人工智能的网络异常检测与诊断系统，采用两阶段模型架构，旨在高效地识别未知异常并对已知异常进行分类。

## 模型架构

系统核心是一个两阶段的检测流程，结合了无监督学习和有监督学习的优点。

### 第一阶段：无监督异常检测 (Autoencoder)

此阶段的目标是快速判断当前网络流量是否"正常"。

*   **模型**: 深度自编码器 (Deep Autoencoder)
*   **原理**:
    1.  仅使用**正常**的网络流量数据对自编码器进行训练。
    2.  模型学习如何将正常数据压缩（编码）并完美地解压缩（解码）。
    3.  当**异常**数据输入时，由于其模式与正常数据不同，模型无法有效重构它，从而产生巨大的**重构误差**。
    4.  通过设定一个合理的误差阈值，任何超过该阈值的数据点都将被标记为异常，并进入下一阶段进行分析。
*   **相关文件**:
    *   模型定义: `src/ai_models/autoencoder_model.py`
    *   训练数据: `data/normal_traffic.csv`
    *   训练脚本: `scripts/train_model.py autoencoder ...`
    *   模型产物: `models/autoencoder_model/` (TensorFlow SavedModel 格式), `models/autoencoder_scaler.pkl`

### 第二阶段：有监督异常分类 (Random Forest)

在确定流量为异常后，此阶段的目标是识别出它具体属于哪种已知的异常类型。

*   **模型**: 随机森林分类器 (Random Forest Classifier)
*   **原理**:
    1.  使用一个包含多种**已标记**异常类型的数据集进行训练。
    2.  随机森林由多棵决策树构成，每棵树都会对输入的异常样本进行分类预测。
    3.  最终的分类结果由所有决策树投票决定，显著提高了分类的准确性和稳定性。
*   **相关文件**:
    *   模型定义: `src/ai_models/error_classifier.py`
    *   训练数据: `data/labeled_anomalies.csv`
    *   训练脚本: `scripts/train_model.py classifier ...`
    *   模型产物: `models/error_classifier.pkl`

## 工作流程

从数据准备到最终检测的完整工作流程如下：

1.  **生成数据**: 运行 `scripts/generate_initial_data.py` 生成用于训练的 `normal_traffic.csv` 和 `labeled_anomalies.csv` 文件。
2.  **训练模型**:
    *   首先，使用正常流量数据训练自编码器，使其掌握正常模式。
    *   然后，使用带标签的异常数据训练随机森林分类器，使其能够分辨不同异常类型。
3.  **检测与诊断**:
    *   在 `scripts/interactive_tester.py` 或其他应用中，新数据首先被送入**自编码器**。
    *   如果重构误差低于阈值，判定为**正常**。
    *   如果重构误差高于阈值，判定为**异常**，并将该数据送入**随机森林分类器**。
    *   分类器输出具体的异常类别，完成诊断。

## 如何使用

请遵循以下步骤来运行整个系统。

### 1. 环境设置

首先，请确保您已经创建并激活了 Python 虚拟环境，并根据 `requirements.txt` 文件安装了所有依赖项。该文件包含了运行此项目所需的所有库（例如 `tensorflow` 和 `scikit-learn`）及其兼容版本。

```bash
# (推荐使用 Python 3.8 或更高版本)
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# 对于 Windows:
# .\venv\Scripts\activate
# 对于 Linux, macOS, 或 WSL:
source venv/bin/activate

# 安装所有依赖
pip install -r requirements.txt
```

### 2. 生成训练数据

运行以下命令来创建初始训练数据：

```bash
python3 scripts/generate_initial_data.py
```

### 3. 训练模型

必须按顺序训练模型：

**a. 训练自编码器**

```bash
python3 scripts/train_model.py autoencoder --data_path data/normal_traffic.csv
```

**b. 训练错误分类器**

在训练分类器前，请确保 `config/system_config.json` 文件中的 `classifier.classes` 列表包含了 `data/labeled_anomalies.csv` 中所有独特的 `label` 值。

```bash
python3 scripts/train_model.py classifier --data_path data/labeled_anomalies.csv
```

### 4. 运行交互式测试

使用以下命令启动交互式终端，以测试模型的检测效果。您可以手动输入各项指标，或直接按回车以使用默认的正常值。

```bash
python3 scripts/interactive_tester.py
```

您也可以使用 `--auto` 标志以非交互模式运行一次自动检测：

```bash
python3 scripts/interactive_tester.py --auto
```

## 文件结构说明

*   `config/system_config.json`: 核心配置文件，定义模型参数和路径。
*   `data/`: 存放原始数据、生成数据和模拟输入。
*   `models/`: 存放所有训练完成的模型文件。
*   `scripts/`: 存放各类操作脚本（数据生成、训练、测试等）。
*   `src/`: 存放项目的主要源代码。
    *   `ai_models/`: 包含两个核心模型的类定义。
    *   `anomaly_detector/`: 包含整合模型的异常检测引擎。
    *   `...` (其他辅助模块)
*   `README.md`: 本文档。 