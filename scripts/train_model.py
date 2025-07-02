#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI模型训练脚本

本脚本用于训练AI网络异常检测系统的核心模型，包括：
1. 自编码器（Autoencoder）：用于无监督学习正常网络流量模式。
2. 错误分类器（Error Classifier）：用于有监督学习，对已知的异常类型进行分类。

使用方法：
- 训练自编码器:
  python scripts/train_model.py autoencoder --data_path data/normal_traffic.csv

- 训练错误分类器:
  python scripts/train_model.py classifier --data_path data/labeled_anomalies.csv
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger.system_logger import SystemLogger
from src.ai_models.autoencoder_model import AutoencoderModel
from src.ai_models.error_classifier import ErrorClassifier

def load_config(config_path="config/system_config.json"):
    """加载系统配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{config_path}' 未找到。请确保该文件存在。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{config_path}' 格式不正确。")
        sys.exit(1)

def train_autoencoder(config: dict, logger: SystemLogger, data_path: str):
    """训练自编码器模型"""
    logger.info("开始训练自编码器模型...")
    
    try:
        # 加载数据
        logger.info(f"从 '{data_path}' 加载正常流量数据...")
        training_data = pd.read_csv(data_path)
        logger.info(f"成功加载 {len(training_data)} 条数据。")
        
        # 初始化模型
        autoencoder = AutoencoderModel(config['ai_models']['autoencoder'], logger)
        
        # 训练模型
        result = autoencoder.train(training_data.values, epochs=config['ai_models']['autoencoder'].get('epochs', 100))
        
        logger.info("自编码器模型训练完成。")
        logger.info(f"训练结果: {result}")
        
    except FileNotFoundError:
        logger.error(f"数据文件 '{data_path}' 未找到。请提供正确的路径。")
    except Exception as e:
        logger.error(f"自编码器训练过程中发生错误: {e}")

def train_classifier(config: dict, logger: SystemLogger, data_path: str):
    """训练错误分类器模型（仅用于异常类型分类，不包含正常数据）"""
    logger.info("开始训练错误分类器模型...")
    logger.info("注意：此分类器仅用于异常类型分类，正常/异常判断由自编码器负责")
    
    try:
        # 加载异常数据
        logger.info(f"从 '{data_path}' 加载带标签的异常数据...")
        labeled_data = pd.read_csv(data_path)
        logger.info(f"成功加载 {len(labeled_data)} 条异常数据。")
        
        if 'label' not in labeled_data.columns:
            logger.error(f"数据文件 '{data_path}' 中缺少 'label' 列。")
            return
            
        # 分离特征和标签
        features = labeled_data.drop('label', axis=1)
        labels = labeled_data['label']

        # 清洗标签数据，去除首尾空格
        cleaned_labels = labels.str.strip()

        # 显示类别分布
        label_counts = cleaned_labels.value_counts()
        logger.info("异常类别分布：")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} 条")

        # 将数据划分为训练集和测试集 (80/20)
        # 使用 stratify 确保在切分后，训练集和测试集中的类别分布与原始数据集一致
        X_train, X_test, y_train, y_test = train_test_split(
            features, cleaned_labels, test_size=0.2, random_state=42, stratify=cleaned_labels
        )
        logger.info(f"数据集划分完成。训练集: {len(X_train)} 条, 测试集: {len(y_test)} 条。")
        
        # 更新配置中的类别列表
        classifier_config = config['ai_models']['classifier'].copy()
        unique_labels = sorted(cleaned_labels.unique().tolist())
        classifier_config['classes'] = unique_labels
        logger.info(f"分类器支持的异常类别: {unique_labels}")
        
        # 初始化模型
        classifier = ErrorClassifier(classifier_config, logger)
        
        # 训练模型 (仅使用训练集)
        logger.info("开始使用训练集训练模型...")
        train_result = classifier.train(X_train.values, y_train.tolist())
        logger.info(f"模型训练完成。训练集准确率: {train_result.get('accuracy', 'N/A'):.3f}")

        # 在测试集上评估模型性能
        logger.info("开始使用测试集评估模型性能...")
        
        # 需要从分类器内部获取已经训练好的模型和标签编码器
        trained_model = classifier.classifier
        label_encoder = classifier.label_encoder

        if trained_model is None:
            logger.error("模型未成功训练，无法进行评估。")
            return

        # 进行预测
        predictions_encoded = trained_model.predict(X_test.values)
        predictions = label_encoder.inverse_transform(predictions_encoded)
        
        # 计算并打印评估指标
        accuracy = accuracy_score(y_test, predictions)
        
        # 获取测试数据中实际存在的标签
        unique_labels_in_test = sorted(y_test.unique())
        
        # 生成分类报告时，明确指定标签和对应的名称
        report = classification_report(
            y_test, 
            predictions, 
            labels=unique_labels_in_test,
            target_names=unique_labels_in_test
        )
        
        logger.info("错误分类器模型评估完成。")
        logger.info(f"测试集准确率 (Accuracy): {accuracy:.3f}")
        logger.info("详细分类报告 (Classification Report):\n" + report)

    except FileNotFoundError:
        logger.error(f"数据文件 '{data_path}' 未找到。请提供正确的路径。")
        print(f"数据文件 '{data_path}' 未找到。请提供正确的路径。")
    except Exception as e:
        logger.error(f"分类器训练过程中发生错误: {e}", exc_info=True)
        print(f"分类器训练过程中发生了一个错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数，解析参数并启动训练"""
    parser = argparse.ArgumentParser(description="AI模型训练脚本")
    parser.add_argument(
        "model_type",
        choices=['autoencoder', 'classifier'],
        help="要训练的模型类型: 'autoencoder' 或 'classifier'"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="训练数据的CSV文件路径"
    )
    parser.add_argument(
        "--config",
        default="config/system_config.json",
        help="系统配置文件的路径"
    )
    
    args = parser.parse_args()
    
    # 加载配置和初始化日志
    config = load_config(args.config)
    logger = SystemLogger(config['logging'])
    
    logger.info(f"开始训练 '{args.model_type}' 模型...")
    
    if args.model_type == 'autoencoder':
        train_autoencoder(config, logger, args.data_path)
    elif args.model_type == 'classifier':
        train_classifier(config, logger, args.data_path)
        
    logger.info("训练脚本执行完毕。")

if __name__ == "__main__":
    main() 