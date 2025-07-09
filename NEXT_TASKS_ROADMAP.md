# 🚀 后续任务路线图：从DLC到生产部署

## 📋 当前状态

✅ **已完成**:
- DLC模型文件 (247.9KB, 生产就绪)
- 完整的训练和测试框架
- 详细的技术文档
- 性能验证和鲁棒性测试

🎯 **下一步目标**: 在目标板子上实现实时网络异常检测

---

## 🛣️ 分阶段任务规划

### 📅 Phase 1: 基础数据采集 (优先级：🔥 高)
**时间估计**: 1-2周  
**目标**: 建立11维网络监控数据的实时采集能力

#### Task 1.1: WiFi信号监控模块
**技术挑战**: 
- 不同Linux系统的WiFi接口差异
- 实时性要求vs系统调用开销
- 信号断连时的异常处理

**实现要点**:
```c
// 示例：WiFi信号质量获取
struct iwreq wreq;
int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

// 获取信号质量
strcpy(wreq.ifr_name, "wlan0");
ioctl(sockfd, SIOCGIWSTATS, &wreq);
wireless_quality = wreq.u.qual.qual;
signal_level = wreq.u.qual.level;
noise_level = wreq.u.qual.noise;
```

**详细讲解**:
1. **接口选择**: 优先使用`nl80211`接口（现代），回退到`iwconfig`（兼容）
2. **数据平滑**: 使用滑动窗口平均值减少瞬时波动
3. **异常处理**: WiFi断开时使用默认值或上次有效值
4. **性能优化**: 缓存socket描述符，避免重复创建

#### Task 1.2: 网络流量统计模块
**技术挑战**:
- 计数器溢出和重置处理
- 多网卡环境的接口选择
- 增量计算的准确性

**实现要点**:
```c
// 读取网络统计 /proc/net/dev
FILE *fp = fopen("/proc/net/dev", "r");
char line[256];
while (fgets(line, sizeof(line), fp)) {
    if (strstr(line, "wlan0:")) {
        sscanf(line, "%*s %lu %lu %*d %*d %*d %*d %*d %*d %lu %lu",
               &rx_bytes, &rx_packets, &tx_bytes, &tx_packets);
        break;
    }
}
```

**详细讲解**:
1. **数据源**: `/proc/net/dev` 提供实时网络统计
2. **增量计算**: current_value - previous_value = delta
3. **溢出检测**: 如果delta < 0，说明计数器重置
4. **速率计算**: delta / time_interval = rate

#### Task 1.3: 网络延迟测试模块
**技术挑战**:
- 非阻塞ping实现
- DNS解析超时控制
- 网络不可达时的降级策略

**实现要点**:
```c
// 非阻塞ping实现
int create_ping_socket() {
    int sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    struct timeval timeout = {1, 0}; // 1秒超时
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    return sockfd;
}

// DNS解析时间测量
double measure_dns_resolution(const char* hostname) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    struct hostent *he = gethostbyname(hostname);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) * 1000.0 + 
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}
```

**详细讲解**:
1. **ping实现**: 使用ICMP协议，需要root权限或特殊配置
2. **DNS测试**: 测试到8.8.8.8或其他公共DNS的解析时间
3. **超时处理**: 设置合理超时避免阻塞主循环
4. **错误降级**: 网络不可达时使用预设的高延迟值

#### Task 1.4: 系统资源监控模块
**技术挑战**:
- 多核CPU的平均使用率计算
- 内存统计的准确性
- 高频采集对系统的影响

**实现要点**:
```c
// CPU使用率计算
typedef struct {
    long user, nice, system, idle, iowait, irq, softirq;
} cpu_stat_t;

double calculate_cpu_usage(cpu_stat_t *prev, cpu_stat_t *curr) {
    long prev_total = prev->user + prev->nice + prev->system + prev->idle;
    long curr_total = curr->user + curr->nice + curr->system + curr->idle;
    long total_diff = curr_total - prev_total;
    long idle_diff = curr->idle - prev->idle;
    
    return 100.0 * (total_diff - idle_diff) / total_diff;
}

// 内存使用率计算
double get_memory_usage() {
    FILE *fp = fopen("/proc/meminfo", "r");
    long mem_total = 0, mem_available = 0;
    char line[128];
    
    while (fgets(line, sizeof(line), fp)) {
        sscanf(line, "MemTotal: %ld kB", &mem_total);
        sscanf(line, "MemAvailable: %ld kB", &mem_available);
    }
    
    return 100.0 * (mem_total - mem_available) / mem_total;
}
```

**详细讲解**:
1. **CPU计算**: 基于`/proc/stat`的差值计算，避免瞬时读数
2. **内存计算**: 使用MemAvailable而非MemFree，更准确反映可用内存
3. **采集频率**: 建议1-5秒间隔，平衡实时性和系统负载
4. **数据校验**: 检查数值合理性，避免异常值影响推理

---

### 📅 Phase 2: SNPE运行时集成 (优先级：🔥 高)
**时间估计**: 1-2周  
**目标**: 在目标硬件上成功加载和运行DLC模型

#### Task 2.1: SNPE环境配置
**技术挑战**:
- 不同ARM架构的库兼容性
- 动态链接库路径配置
- 加速器(CPU/GPU/DSP)检测

**实现要点**:
```bash
# SNPE环境设置
export SNPE_ROOT=/opt/snpe-2.26.2.240911
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/aarch64-linux-gcc9.4:$LD_LIBRARY_PATH
export PATH=$SNPE_ROOT/bin/aarch64-linux-gcc9.4:$PATH

# 检查可用的计算单元
snpe-platform-validator --runtime cpu
snpe-platform-validator --runtime gpu  
snpe-platform-validator --runtime dsp
```

**详细讲解**:
1. **库依赖**: 确保SNPE运行时库与目标系统ABI兼容
2. **权限配置**: DSP运行可能需要特殊权限或设备节点
3. **性能基准**: 不同计算单元的性能和功耗特性
4. **故障排除**: 常见的库加载失败和解决方案

#### Task 2.2: DLC模型加载管理
**技术挑战**:
- 两个模型的协调加载
- 内存优化和模型缓存
- 加载失败的降级策略

**实现要点**:
```cpp
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"

class ModelManager {
private:
    std::unique_ptr<zdl::SNPE::SNPE> detector_snpe;
    std::unique_ptr<zdl::SNPE::SNPE> classifier_snpe;
    
public:
    bool loadModels(const std::string& detector_dlc, 
                   const std::string& classifier_dlc) {
        // 加载异常检测模型
        detector_snpe = zdl::SNPE::SNPEFactory::newSnpe()
            .setDlc(detector_dlc)
            .setRuntimeProcessor(zdl::DlSystem::Runtime_t::CPU)
            .build();
            
        // 加载异常分类模型
        classifier_snpe = zdl::SNPE::SNPEFactory::newSnpe()
            .setDlc(classifier_dlc)
            .setRuntimeProcessor(zdl::DlSystem::Runtime_t::CPU)
            .build();
            
        return detector_snpe && classifier_snpe;
    }
};
```

**详细讲解**:
1. **模型验证**: 加载后验证输入输出维度和数据类型
2. **内存管理**: 合理分配内存缓冲区，避免频繁分配释放
3. **并发安全**: 多线程环境下的模型访问同步
4. **错误处理**: 详细的错误日志和恢复机制

#### Task 2.3: 推理接口封装
**技术挑战**:
- 输入数据格式转换
- 两阶段推理的协调
- 输出结果的解析和后处理

**实现要点**:
```cpp
struct InferenceResult {
    bool is_anomaly;
    std::string anomaly_type;
    float detection_confidence;
    float classification_confidence;
};

class InferenceEngine {
public:
    InferenceResult predict(const std::vector<float>& input_data) {
        // 阶段1：异常检测
        auto detection_output = runDetection(input_data);
        bool is_anomaly = detection_output[1] > detection_output[0];
        float detection_conf = std::max(detection_output[0], detection_output[1]);
        
        InferenceResult result;
        result.is_anomaly = is_anomaly;
        result.detection_confidence = detection_conf;
        
        if (is_anomaly) {
            // 阶段2：异常分类
            auto classification_output = runClassification(input_data);
            int max_idx = std::max_element(classification_output.begin(), 
                                         classification_output.end()) - 
                         classification_output.begin();
            
            result.anomaly_type = anomaly_types[max_idx];
            result.classification_confidence = classification_output[max_idx];
        }
        
        return result;
    }
};
```

**详细讲解**:
1. **数据流**: 11维输入 → 标准化 → 两阶段推理 → 结果输出
2. **性能优化**: 输入缓冲区复用，减少内存分配
3. **精度处理**: 使用float32确保精度和兼容性
4. **异常处理**: 推理失败时的错误码和恢复策略

---

### 📅 Phase 3: 数据处理管道 (优先级：⚡ 中高)
**时间估计**: 1周  
**目标**: 构建稳定的数据预处理和标准化流程

#### Task 3.1: 标准化器集成
**技术挑战**:
- Python pkl文件在C++中的解析
- 标准化参数的精确应用
- 数值精度的保持

**实现要点**:
```cpp
class DataScaler {
private:
    std::vector<float> mean_values;
    std::vector<float> scale_values;
    
public:
    bool loadScaler(const std::string& scaler_file) {
        // 方案1：使用Python C API加载pkl文件
        // 方案2：将scaler参数导出为JSON/CSV格式
        // 方案3：直接在代码中硬编码参数
        
        // 推荐方案2：JSON格式便于维护
        std::ifstream file(scaler_file);
        nlohmann::json scaler_params;
        file >> scaler_params;
        
        mean_values = scaler_params["mean"];
        scale_values = scaler_params["scale"];
        return true;
    }
    
    std::vector<float> transform(const std::vector<float>& raw_data) {
        std::vector<float> scaled_data(raw_data.size());
        for (size_t i = 0; i < raw_data.size(); ++i) {
            scaled_data[i] = (raw_data[i] - mean_values[i]) / scale_values[i];
        }
        return scaled_data;
    }
};
```

**详细讲解**:
1. **格式转换**: 建议将pkl参数导出为JSON格式便于C++解析
2. **精度保证**: 使用与训练时相同的float32精度
3. **参数验证**: 检查mean和scale数组的长度和合理性
4. **性能优化**: 预分配内存避免动态分配

#### Task 3.2: 数据验证和清洗
**技术挑战**:
- 异常值检测和处理
- 缺失数据的插值策略
- 数据质量评分

**实现要点**:
```cpp
class DataValidator {
public:
    struct ValidationResult {
        bool is_valid;
        std::vector<bool> feature_validity;
        float quality_score;
    };
    
    ValidationResult validate(const std::vector<float>& data) {
        ValidationResult result;
        result.feature_validity.resize(data.size());
        
        // 定义每个特征的合理范围
        std::vector<std::pair<float, float>> valid_ranges = {
            {0, 100},       // wlan0_wireless_quality
            {-100, -10},    // wlan0_signal_level
            {-100, -30},    // wlan0_noise_level
            {0, 1e8},       // wlan0_rx_packets
            {0, 1e8},       // wlan0_tx_packets
            {0, 1e10},      // wlan0_rx_bytes
            {0, 1e10},      // wlan0_tx_bytes
            {0, 5000},      // gateway_ping_time
            {0, 5000},      // dns_resolution_time
            {0, 100},       // memory_usage_percent
            {0, 100}        // cpu_usage_percent
        };
        
        int valid_count = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            bool is_feature_valid = (data[i] >= valid_ranges[i].first && 
                                   data[i] <= valid_ranges[i].second);
            result.feature_validity[i] = is_feature_valid;
            if (is_feature_valid) valid_count++;
        }
        
        result.quality_score = float(valid_count) / data.size();
        result.is_valid = result.quality_score >= 0.8; // 80%有效性阈值
        
        return result;
    }
};
```

**详细讲解**:
1. **范围检查**: 基于物理含义设定合理的数值范围
2. **缺失处理**: 使用历史平均值或上次有效值填充
3. **异常标记**: 标记异常值但不直接丢弃，保留用于调试
4. **质量评分**: 提供数据质量的量化指标

---

### 📅 Phase 4: 实时推理引擎 (优先级：⚡ 中高)
**时间估计**: 2周  
**目标**: 构建高效稳定的多线程推理系统

#### Task 4.1: 多线程架构设计
**技术挑战**:
- 线程间数据同步
- 任务队列的设计
- 线程池的管理

**实现要点**:
```cpp
class RealtimeInferenceEngine {
private:
    std::thread data_collection_thread;
    std::thread inference_thread;
    std::thread result_processing_thread;
    
    std::queue<std::vector<float>> data_queue;
    std::queue<InferenceResult> result_queue;
    std::mutex data_mutex, result_mutex;
    std::condition_variable data_cv, result_cv;
    
public:
    void start() {
        data_collection_thread = std::thread(&RealtimeInferenceEngine::dataCollectionLoop, this);
        inference_thread = std::thread(&RealtimeInferenceEngine::inferenceLoop, this);
        result_processing_thread = std::thread(&RealtimeInferenceEngine::resultProcessingLoop, this);
    }
    
private:
    void dataCollectionLoop() {
        while (running) {
            auto data = collectNetworkData(); // 调用数据采集模块
            
            std::lock_guard<std::mutex> lock(data_mutex);
            data_queue.push(data);
            data_cv.notify_one();
            
            std::this_thread::sleep_for(std::chrono::seconds(5)); // 5秒间隔
        }
    }
    
    void inferenceLoop() {
        while (running) {
            std::unique_lock<std::mutex> lock(data_mutex);
            data_cv.wait(lock, [this] { return !data_queue.empty() || !running; });
            
            if (!running) break;
            
            auto data = data_queue.front();
            data_queue.pop();
            lock.unlock();
            
            auto result = inference_engine.predict(data); // 执行推理
            
            std::lock_guard<std::mutex> result_lock(result_mutex);
            result_queue.push(result);
            result_cv.notify_one();
        }
    }
};
```

**详细讲解**:
1. **线程职责**:
   - 数据采集线程：持续采集11维网络数据
   - 推理线程：执行模型推理
   - 结果处理线程：处理异常告警和日志
2. **同步机制**: 使用条件变量避免忙等待
3. **队列管理**: 限制队列大小防止内存溢出
4. **优雅退出**: 支持线程的优雅停止和资源清理

#### Task 4.2: 性能监控系统
**技术挑战**:
- 实时性能指标统计
- 内存和CPU使用监控
- 性能瓶颈识别

**实现要点**:
```cpp
class PerformanceMonitor {
private:
    struct Metrics {
        double avg_inference_time;
        double max_inference_time;
        size_t total_inferences;
        size_t failed_inferences;
        double memory_usage_mb;
        double cpu_usage_percent;
    };
    
public:
    void recordInference(double inference_time, bool success) {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        
        if (success) {
            metrics.total_inferences++;
            updateAverageTime(inference_time);
            metrics.max_inference_time = std::max(metrics.max_inference_time, inference_time);
        } else {
            metrics.failed_inferences++;
        }
    }
    
    void logMetrics() {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        
        std::cout << "=== Performance Metrics ===" << std::endl;
        std::cout << "Total inferences: " << metrics.total_inferences << std::endl;
        std::cout << "Failed inferences: " << metrics.failed_inferences << std::endl;
        std::cout << "Average inference time: " << metrics.avg_inference_time << " ms" << std::endl;
        std::cout << "Max inference time: " << metrics.max_inference_time << " ms" << std::endl;
        std::cout << "Memory usage: " << metrics.memory_usage_mb << " MB" << std::endl;
        std::cout << "CPU usage: " << metrics.cpu_usage_percent << "%" << std::endl;
    }
};
```

**详细讲解**:
1. **关键指标**:
   - 推理延迟：单次推理耗时
   - 吞吐量：每秒处理的样本数
   - 成功率：推理成功的比例
   - 资源占用：内存和CPU使用率
2. **监控频率**: 实时记录，定期汇总输出
3. **告警机制**: 性能指标异常时触发告警
4. **历史数据**: 保存历史性能数据用于趋势分析

---

### 📅 Phase 5: 系统集成和API (优先级：🛠️ 中等)
**时间估计**: 1-2周  
**目标**: 提供标准化的API接口和系统集成能力

#### Task 5.1: RESTful API设计
**技术挑战**:
- API接口的设计和文档
- 并发请求处理
- 数据格式标准化

**实现要点**:
```cpp
// 使用cpp-httplib或类似库
#include <httplib.h>
#include <nlohmann/json.hpp>

class APIServer {
public:
    void start(int port = 8080) {
        httplib::Server server;
        
        // 获取当前状态
        server.Get("/api/v1/status", [this](const httplib::Request&, httplib::Response& res) {
            nlohmann::json status = {
                {"status", "running"},
                {"uptime", getUptime()},
                {"total_inferences", performance_monitor.getTotalInferences()},
                {"last_result", getLastResult()}
            };
            res.set_content(status.dump(), "application/json");
        });
        
        // 手动推理接口
        server.Post("/api/v1/predict", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto json_data = nlohmann::json::parse(req.body);
                std::vector<float> input_data = json_data["data"];
                
                if (input_data.size() != 11) {
                    res.status = 400;
                    res.set_content("{\"error\": \"Input data must have 11 dimensions\"}", "application/json");
                    return;
                }
                
                auto result = inference_engine.predict(input_data);
                
                nlohmann::json response = {
                    {"is_anomaly", result.is_anomaly},
                    {"anomaly_type", result.anomaly_type},
                    {"detection_confidence", result.detection_confidence},
                    {"classification_confidence", result.classification_confidence},
                    {"timestamp", getCurrentTimestamp()}
                };
                
                res.set_content(response.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content("{\"error\": \"Internal server error\"}", "application/json");
            }
        });
        
        server.listen("0.0.0.0", port);
    }
};
```

**详细讲解**:
1. **API端点设计**:
   - `/api/v1/status` - 系统状态查询
   - `/api/v1/predict` - 手动推理接口
   - `/api/v1/metrics` - 性能指标查询
   - `/api/v1/config` - 配置管理接口
2. **数据格式**: 使用JSON标准化输入输出
3. **错误处理**: 详细的HTTP状态码和错误信息
4. **并发支持**: 支持多个客户端同时访问

#### Task 5.2: WebSocket实时推送
**技术挑战**:
- 实时数据推送
- 连接管理和心跳检测
- 数据流控制

**实现要点**:
```cpp
class WebSocketServer {
private:
    std::set<websocketpp::connection_hdl> connections;
    std::mutex connections_mutex;
    
public:
    void broadcastResult(const InferenceResult& result) {
        nlohmann::json message = {
            {"type", "inference_result"},
            {"timestamp", getCurrentTimestamp()},
            {"data", {
                {"is_anomaly", result.is_anomaly},
                {"anomaly_type", result.anomaly_type},
                {"detection_confidence", result.detection_confidence},
                {"classification_confidence", result.classification_confidence}
            }}
        };
        
        std::lock_guard<std::mutex> lock(connections_mutex);
        for (auto hdl : connections) {
            server.send(hdl, message.dump(), websocketpp::frame::opcode::text);
        }
    }
    
    void onOpen(websocketpp::connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connections_mutex);
        connections.insert(hdl);
    }
    
    void onClose(websocketpp::connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connections_mutex);
        connections.erase(hdl);
    }
};
```

**详细讲解**:
1. **实时推送**: 检测结果实时推送给连接的客户端
2. **连接管理**: 维护活跃连接列表，自动清理断开连接
3. **消息格式**: 标准化的JSON消息格式
4. **流控机制**: 避免推送过于频繁导致客户端压力

---

### 📅 Phase 6: 部署优化和维护 (优先级：🔧 长期)
**时间估计**: 持续优化  
**目标**: 提高系统稳定性和可维护性

#### Task 6.1: 容器化部署
**技术挑战**:
- ARM架构的Docker镜像构建
- SNPE库的容器化打包
- 设备权限和网络访问

**实现要点**:
```dockerfile
# Dockerfile for ARM64
FROM arm64v8/ubuntu:22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libssl-dev \
    wireless-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# 复制SNPE库
COPY snpe-2.26.2.240911 /opt/snpe/
ENV SNPE_ROOT=/opt/snpe
ENV LD_LIBRARY_PATH=${SNPE_ROOT}/lib/aarch64-linux-gcc9.4:$LD_LIBRARY_PATH

# 复制应用程序
COPY inference_engine /app/
COPY realistic_end_to_end_*.dlc /app/models/
COPY realistic_raw_data_scaler.json /app/models/

WORKDIR /app
CMD ["./inference_engine"]
```

**详细讲解**:
1. **多阶段构建**: 分离编译环境和运行环境
2. **依赖管理**: 精确控制运行时依赖
3. **配置外部化**: 使用环境变量和配置文件
4. **健康检查**: 容器健康状态监控

#### Task 6.2: 远程监控系统
**技术挑战**:
- 设备状态的远程收集
- 网络连接的可靠性
- 数据安全和隐私保护

**实现要点**:
```cpp
class RemoteMonitoring {
private:
    std::string server_url;
    std::string device_id;
    
public:
    void reportStatus() {
        nlohmann::json report = {
            {"device_id", device_id},
            {"timestamp", getCurrentTimestamp()},
            {"system_info", {
                {"cpu_usage", getCurrentCPUUsage()},
                {"memory_usage", getCurrentMemoryUsage()},
                {"disk_usage", getCurrentDiskUsage()},
                {"network_status", getNetworkStatus()}
            }},
            {"inference_stats", {
                {"total_inferences", performance_monitor.getTotalInferences()},
                {"error_rate", performance_monitor.getErrorRate()},
                {"avg_latency", performance_monitor.getAverageLatency()}
            }},
            {"recent_anomalies", getRecentAnomalies()}
        };
        
        // 发送到监控服务器
        httplib::Client client(server_url);
        auto res = client.Post("/api/devices/report", report.dump(), "application/json");
    }
};
```

**详细讲解**:
1. **监控指标**: 系统状态、推理性能、异常统计
2. **上报策略**: 定期上报 + 异常触发上报
3. **数据压缩**: 大量数据时使用压缩减少传输
4. **离线缓存**: 网络断开时本地缓存数据

---

## 🎯 关键里程碑

### 🏁 Milestone 1: 基础功能验证 (2周后)
**验收标准**:
- [ ] 11维数据能够实时采集
- [ ] DLC模型能够成功加载
- [ ] 端到端推理流程能够运行
- [ ] 基础性能指标达标

### 🏁 Milestone 2: 系统集成完成 (4周后)
**验收标准**:
- [ ] 多线程系统稳定运行
- [ ] API接口功能完备
- [ ] 性能监控系统就位
- [ ] 基础异常处理机制完善

### 🏁 Milestone 3: 生产部署就绪 (6-8周后)
**验收标准**:
- [ ] 7x24小时稳定运行验证
- [ ] 远程监控和维护能力
- [ ] 完整的部署和运维文档
- [ ] 性能和准确性达到生产要求

---

## ⚠️ 风险评估和缓解策略

### 🔴 高风险项
1. **SNPE库兼容性问题**
   - 风险: 目标硬件不支持特定SNPE版本
   - 缓解: 提前验证，准备多个SNPE版本

2. **实时性能要求**
   - 风险: 推理延迟超过业务要求
   - 缓解: 性能基准测试，优化关键路径

### 🟡 中等风险项
1. **网络环境复杂性**
   - 风险: 不同网络环境下数据采集差异
   - 缓解: 充分的环境测试，自适应配置

2. **长期稳定性**
   - 风险: 内存泄漏或资源耗尽
   - 缓解: 压力测试，完善的监控告警

---

## 📚 技术资源和参考

### 📖 必读文档
1. **SNPE Developer Guide** - SNPE集成和优化
2. **Linux Network Programming** - 网络数据采集
3. **Multithreading in C++** - 多线程架构设计

### 🛠️ 推荐工具
1. **开发调试**: GDB, Valgrind, strace
2. **性能分析**: perf, htop, iotop
3. **网络测试**: iperf3, tcpdump, wireshark

### 📦 依赖库
1. **HTTP服务**: cpp-httplib 或 Beast
2. **JSON处理**: nlohmann/json
3. **WebSocket**: websocketpp
4. **日志记录**: spdlog

---

**🎯 总结**: 这个路线图将指导您从当前的DLC文件状态，逐步构建一个完整的生产级网络异常检测系统。每个阶段都有明确的目标和验收标准，确保项目的可控推进。 