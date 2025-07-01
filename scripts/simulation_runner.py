import os
import sys
import json
import numpy as np

# 将src目录添加到Python路径中，以便导入项目模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_processor.feature_extractor import FeatureExtractor
# 使用正确的类名并设置别名，以减少代码改动
from anomaly_detector.anomaly_engine import AnomalyDetectionEngine as AnomalyEngine
from logger.system_logger import SystemLogger
from ai_models.autoencoder_model import AutoencoderModel
from ai_models.error_classifier import ErrorClassifier

def load_config():
    """加载系统配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_simulation():
    """
    运行一个模拟，使用预定义的输入数据来测试AI引擎。
    """
    print("--- 开始AI异常检测模拟 ---")

    # 1. 初始化系统组件
    config = load_config()
    logger = SystemLogger(config['logging'])
    feature_extractor = FeatureExtractor(config['data_collection']['metrics'], logger)

    # 加载真实的AI模型
    logger.info("正在加载真实的AI模型用于模拟...")
    autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
    error_classifier = ErrorClassifier(config['ai_models']['classifier'], logger)
    
    # 将真实的模型实例注入到引擎中
    anomaly_engine = AnomalyEngine(
        config=config['anomaly_detection'], 
        autoencoder=autoencoder,
        error_classifier=error_classifier,
        buffer_manager=None, # 模拟器中不使用缓冲区
        logger=logger
    )

    print("系统组件和真实AI模型初始化完成。")

    # 2. 从文件加载模拟输入数据
    inputs_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation_inputs.json')
    try:
        with open(inputs_path, 'r', encoding='utf-8') as f:
            simulation_inputs = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到模拟输入文件 {inputs_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析JSON文件 {inputs_path}")
        return

    # 3. 运行检测流程
    for test_case in simulation_inputs:
        case_name = test_case.get("name", "未命名测试")
        raw_input = test_case.get("data")

        if not raw_input:
            print(f"跳过无效的测试用例： {case_name}")
            continue
        
        print(f"\n--- 正在测试【{case_name}】输入 ---")
        process_input(feature_extractor, anomaly_engine, raw_input)


    print("\n--- 模拟结束 ---")

def process_input(extractor, engine, raw_input):
    """
    处理单个输入字典，完成特征提取和异常检测。
    """
    print(f"原始输入: {raw_input}")
    
    # 步骤1: 特征提取
    feature_vector = extractor.extract_features(raw_input)
    
    if feature_vector.size == 0:
        print("特征提取失败，跳过检测。")
        return

    # 获取并打印特征向量的详细信息
    feature_names = extractor.get_feature_names()
    print("--- 原始数据 -> 特征向量转换关系 ---")
    if feature_names:
        for name, value in zip(feature_names, feature_vector):
            print(f"{name:<30}: {value:.4f}")
    else:
        print("未能获取特征名称，直接打印向量。")
        print(f"生成的特征向量: {feature_vector}")
    print("------------------------------------")


    # 步骤2: 异常检测
    # 调用为模拟器新增的、直接接收向量的简化方法
    is_anomaly, details = engine.detect_anomaly_from_vector(feature_vector, feature_names)
    
    # 步骤3: 打印结果
    if is_anomaly:
        print(f"\033[91m检测结果: 发现异常!\033[0m")
        print(f"详细信息: {details}")
    else:
        print(f"\033[92m检测结果: 未发现异常。\033[0m")
        print(f"详细信息: {details}")

if __name__ == "__main__":
    run_simulation() 