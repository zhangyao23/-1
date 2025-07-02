import os
import sys
import json
import argparse
import numpy as np

# 将src目录添加到Python路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from anomaly_detector.anomaly_engine import AnomalyDetectionEngine
from logger.system_logger import SystemLogger
from ai_models.autoencoder_model import AutoencoderModel
from ai_models.error_classifier import ErrorClassifier

def load_config():
    """加载系统配置"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_raw_to_6d_features(raw_data):
    """
    直接将11维原始数据转换为6维特征
    映射逻辑基于训练数据的格式
    """
    # 6个核心特征名称（与训练数据一致）
    features = np.zeros(6)
    
    # 1. avg_signal_strength: 基于wireless_quality
    # 训练数据范围：70-90（正常），15-45（异常）
    features[0] = raw_data.get('wlan0_wireless_quality', 70.0)
    
    # 2. avg_data_rate: 基于传输速率，归一化到0-1
    # 训练数据范围：0.45-0.75（正常），0.1-0.5（异常）
    send_rate = raw_data.get('wlan0_send_rate_bps', 500000.0)
    recv_rate = raw_data.get('wlan0_recv_rate_bps', 1500000.0)
    avg_rate = (send_rate + recv_rate) / 2
    # 将速率归一化到0-1范围（假设最大速率为2000000 bps）
    features[1] = min(avg_rate / 2000000.0, 1.0)
    
    # 3. avg_latency: 基于网络延迟
    # 训练数据范围：10-30ms（正常），50-350ms（异常）
    ping_time = raw_data.get('gateway_ping_time', 12.5)
    dns_time = raw_data.get('dns_response_time', 25.0)
    features[2] = (ping_time + dns_time) / 2
    
    # 4. total_packet_loss: 基于丢包率和重传
    # 训练数据范围：0.001-0.05（正常），0.05-0.8（异常）
    packet_loss = raw_data.get('wlan0_packet_loss_rate', 0.01)
    retrans = raw_data.get('tcp_retrans_segments', 5)
    # 优化重传次数转换逻辑：5次重传不应等于5%丢包
    # 正常情况下5次重传约等于0.5%额外丢包
    retrans_loss = min(retrans / 1000.0, 0.05)  # 最多贡献5%丢包率
    features[3] = packet_loss + retrans_loss
    
    # 5. cpu_usage: CPU使用率
    # 训练数据范围：5-30%（正常），60-95%（异常）
    features[4] = raw_data.get('cpu_percent', 15.0)
    
    # 6. memory_usage: 内存使用率
    # 训练数据范围：30-70%（正常），65-95%（异常）
    features[5] = raw_data.get('memory_percent', 45.0)
    
    return features

def get_feature_names():
    """返回6维特征名称"""
    return [
        'avg_signal_strength', 'avg_data_rate', 'avg_latency',
        'total_packet_loss', 'cpu_usage', 'memory_usage'
    ]

def get_default_inputs():
    """从simulation_inputs.json加载"正常"情况作为默认值"""
    inputs_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation_inputs.json')
    try:
        with open(inputs_path, 'r', encoding='utf-8') as f:
            for case in json.load(f):
                if "正常" in case.get("name", ""):
                    return case.get("data", {})
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或损坏，则使用硬编码的后备值
        return {
            'wlan0_wireless_quality': 70.0, 'wlan0_wireless_level': -55.0,
            'wlan0_packet_loss_rate': 0.01, 'wlan0_send_rate_bps': 500000.0,
            'wlan0_recv_rate_bps': 1500000.0, 'tcp_retrans_segments': 5,
            'gateway_ping_time': 12.5, 'dns_response_time': 25.0,
            'tcp_connection_count': 30, 'cpu_percent': 15.0, 'memory_percent': 45.0
        }
    return {}

def run_interactive_session(engine):
    """运行一次交互式会话，获取用户输入并执行检测"""
    print("\n--- 欢迎来到交互式AI检测终端 ---")
    print("请根据提示输入各项指标的数值。直接按回车键将使用括号内的默认值。")
    print("（在任何时候按 Ctrl+C 都可以退出程序）\n")

    default_inputs = get_default_inputs()
    if not default_inputs:
        print("错误：无法加载默认输入值。")
        return

    raw_input_data = {}
    
    for key, default_value in default_inputs.items():
        while True:
            try:
                prompt = f"  -> 请输入 '{key}' ({default_value}): "
                user_value_str = input(prompt)
                
                if not user_value_str:
                    raw_input_data[key] = float(default_value)
                    break
                
                raw_input_data[key] = float(user_value_str)
                break
            except ValueError:
                print("   输入无效，请输入一个数字。请重试。")
            except KeyboardInterrupt:
                print("\n\n操作已取消。")
                raise

    print("\n--- 您输入的完整数据 ---")
    print(json.dumps(raw_input_data, indent=2, ensure_ascii=False))
    print("--------------------------")

    print("\n正在处理数据并进行AI检测...")
    feature_vector = convert_raw_to_6d_features(raw_input_data)
    feature_names = get_feature_names()
    
    if feature_vector.size == 0:
        print("错误：特征提取失败。")
        return

    is_anomaly, details = engine.detect_anomaly_from_vector(feature_vector, feature_names)

    print("\n\n" + "="*12 + " 检测结果 " + "="*12)
    if is_anomaly:
        print("\033[91m状态: 检测到异常!\033[0m")
        predicted_class = details.get('predicted_class', 'N/A')
        confidence = details.get('confidence', 0.0)
        print(f"预测类型: {predicted_class}")
        print(f"置信度: {confidence:.2%}")
    else:
        print("\033[92m状态: 一切正常\033[0m")

    print("\n--- 详细技术信息 ---")
    error = details.get('reconstruction_error', 'N/A')
    threshold = details.get('threshold', 'N/A')
    print(f"模型重构误差: {error:.6f}")
    print(f"模型异常阈值: {threshold:.6f}")
    print("="*36 + "\n")


def run_auto_session(engine):
    """运行一次非交互式的自动化会话"""
    print("\n--- 正在运行自动化AI检测... ---")

    default_inputs = get_default_inputs()
    if not default_inputs:
        print("错误：无法加载默认输入值。")
        return

    print("\n--- 使用的默认数据 ---")
    print(json.dumps(default_inputs, indent=2, ensure_ascii=False))
    print("--------------------------")

    print("\n正在处理数据并进行AI检测...")
    feature_vector = convert_raw_to_6d_features(default_inputs)
    feature_names = get_feature_names()
    
    if feature_vector.size == 0:
        print("错误：特征提取失败。")
        return

    is_anomaly, details = engine.detect_anomaly_from_vector(feature_vector, feature_names)

    print("\n\n" + "="*12 + " 检测结果 " + "="*12)
    if is_anomaly:
        print("\033[91m状态: 检测到异常!\033[0m")
        predicted_class = details.get('predicted_class', 'N/A')
        confidence = details.get('confidence', 0.0)
        print(f"预测类型: {predicted_class}")
        print(f"置信度: {confidence:.2%}")
    else:
        print("\033[92m状态: 一切正常\033[0m")

    print("\n--- 详细技术信息 ---")
    error = details.get('reconstruction_error', 'N/A')
    threshold = details.get('threshold', 'N/A')
    print(f"模型重构误差: {error:.6f}")
    print(f"模型异常阈值: {threshold:.6f}")
    print("="*36 + "\n")


def main():
    """主函数，用于初始化和运行交互式测试器"""
    # --- 添加命令行参数解析 ---
    parser = argparse.ArgumentParser(description="交互式或自动化AI检测终端")
    parser.add_argument(
        '--auto',
        action='store_true',
        help='如果设置此标志，则以非交互式的自动化模式运行一次检测。'
    )
    args = parser.parse_args()
    # --------------------------

    print("--- 正在初始化AI引擎和模型，请稍候... ---")
    try:
        config = load_config()
        logger = SystemLogger(config['logging'])
        
        # 将日志级别设为WARNING，以获得更干净的交互界面
        logger.set_log_level('WARNING')
        
        autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
        classifier = ErrorClassifier(config['ai_models']['classifier'], logger)
        
        engine = AnomalyDetectionEngine(
            config=config['anomaly_detection'],
            autoencoder=autoencoder, error_classifier=classifier,
            buffer_manager=None, logger=logger
        )

        print("--- 初始化完成 ---")

        if args.auto:
            run_auto_session(engine)
        else:
            while True:
                run_interactive_session(engine)

    except KeyboardInterrupt:
        print("\n程序已退出。感谢使用！")
    except Exception as e:
        print(f"\n发生致命错误: {e}")
        print("请检查配置文件和模型文件是否完整。")


if __name__ == "__main__":
    main() 