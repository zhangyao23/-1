# AI引擎模拟与自定义输入指南

本文档旨在指导您如何使用 `scripts/simulation_runner.py` 脚本，以便用您自定义的数据来测试和验证AI异常检测引擎的核心逻辑。

## 1. 为什么需要模拟？

本系统的主要入口 `src/main.py` 是一个完整的服务，它会自动从操作系统采集实时数据。但在开发和测试阶段，我们经常需要：

-   **独立测试AI逻辑**：在不依赖真实硬件和网络环境的情况下，验证AI模型的判断是否准确。
-   **复现特定场景**：精确地构造某种异常（如高延迟、高丢包）的数据，看模型能否识别。
-   **集成新数据源**：如果您希望将此AI引擎用于其他数据源（如解析日志文件、接收消息队列），本脚本提供了一个完美的起点。

`simulation_runner.py` 脚本加载了与主程序完全相同的AI模型和处理逻辑，但将其数据输入端暴露出来，允许您手动提供一个Python字典作为模拟的"实时"数据。

## 2. 如何运行模拟脚本

直接在项目根目录下运行以下命令即可：

```bash
python scripts/simulation_runner.py
```

脚本会自动加载 `data/simulation_inputs.json` 文件中定义的所有测试用例，并依次打印出详细的检测流程和最终结果。

## 3. 提供自定义输入

从现在开始，模拟脚本的数据源已从脚本内部的硬编码字典，转移到了项目根目录下的 `data/simulation_inputs.json` 文件。这使您可以在不修改任何代码的情况下，轻松地添加、删除或修改测试用例。

### 文件结构

该JSON文件是一个包含多个测试用例的列表。每个测试用例都是一个对象，包含两个字段：
-   `name`: (字符串) 测试用例的描述性名称，会显示在模拟输出中。
-   `data`: (对象) 包含所有原始指标的字典，与 `NetworkDataCollector` 采集到的数据结构一致。

### 如何添加新的测试场景

要添加您自己的测试用例，只需打开 `data/simulation_inputs.json` 文件，并仿照现有结构添加一个新的JSON对象到列表中即可。

例如，要模拟一个"丢包率极高"的场景，您可以添加如下内容：
```json
  {
    "name": "高丢包率场景",
    "data": {
      "wlan0_wireless_quality": 70.0,
      "wlan0_wireless_level": -55.0,
      "wlan0_packet_loss_rate": 25.5,
      "wlan0_send_rate_bps": 500000.0,
      "wlan0_recv_rate_bps": 1500000.0,
      "tcp_retrans_segments": 150,
      "gateway_ping_time": 40.0,
      "dns_response_time": 30.0,
      "tcp_connection_count": 30,
      "cpu_percent": 15.0,
      "memory_percent": 45.0
    }
  }
```
添加完毕后，再次运行 `python scripts/simulation_runner.py`，您的新场景就会被自动加载和测试。

### 可用参数详解

以下是 `data` 对象中可用的核心参数及其含义：

| 参数名 | 含义 | 理想趋势 |
|---|---|---|
| `wlan0_wireless_quality` | WiFi信号质量 | 越高越好 |
| `wlan0_wireless_level` | WiFi信号电平 (dBm) | 越接近0越好 |
| `wlan0_packet_loss_rate` | 接口丢包率 (%) | 越低越好 |
| `wlan0_send_rate_bps` | 接口发送速率 (Bits per second) | / |
| `wlan0_recv_rate_bps` | 接口接收速率 (Bits per second) | / |
| `tcp_retrans_segments` | TCP重传报文段数量 | 越少越好 |
| `gateway_ping_time` | 网关Ping延迟 (毫秒) | 越低越好 |
| `dns_response_time` | DNS解析响应时间 (毫秒) | 越低越好 |
| `tcp_connection_count` | TCP连接总数 | / |
| `cpu_percent` | CPU使用率 (%) | 越低越好 |
| `memory_percent` | 内存使用率 (%) | 越低越好 |

## 4. 解读模拟输出

为了方便专家对模型行为进行分析，模拟脚本的输出经过了专门优化。对于每一次模拟测试，您都会看到一个清晰的转换关系表格。

这个表格展示了AI模型最终接收到的、经过标准化处理的**特征向量 (Feature Vector)**，及其与原始输入指标经过复杂计算后生成的**特征名称 (Feature Name)** 的一一对应关系。

### 输出示例

```
--- 原始数据 -> 特征向量转换关系 ---
avg_data_rate                 : -0.7071
avg_latency                   : 1.2247
avg_signal_strength           : -0.7071
cpu_usage                     : 1.2247
global_kurtosis               : 0.0000
global_mean                   : 0.2861
global_skewness               : 0.0000
global_std                    : 0.4516
max_data_rate                 : -0.7071
max_latency                   : 1.2247
max_packet_loss               : 0.7071
... (更多特征)
------------------------------------
```

### 为什么这很重要？

-   **透明度**: 这个表格揭示了AI模型的"所见"。原始数据（如`gateway_ping_time: 150.8`）本身无法直接被模型使用，它必须被转换为一个或多个标准化的特征（如`avg_latency`, `max_latency`等）。
-   **可解释性**: 当模型做出"异常"判断时，您可以回溯这个表格。如果看到 `avg_latency` 或 `cpu_usage` 对应的数值非常高，就能直观地理解模型是基于哪些关键证据做出决策的。这为专家提供了验证和信任模型的基础。 