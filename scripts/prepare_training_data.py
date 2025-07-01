#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集准备脚本

本脚本用于合并正常流量数据和带标签的异常数据，
为分类器模型的训练准备一个统一的数据集。
"""

import pandas as pd
from pathlib import Path

def prepare_data():
    """
    合并数据源并生成统一的训练CSV文件。
    """
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    
    # 定义输入和输出文件路径
    normal_traffic_file = data_path / 'normal_traffic.csv'
    labeled_anomalies_file = data_path / 'labeled_anomalies.csv'
    output_file = data_path / 'combined_training_data.csv'
    
    print("开始处理数据集...")

    try:
        # 1. 加载正常流量数据
        print(f"正在加载正常流量数据: {normal_traffic_file}")
        normal_df = pd.read_csv(normal_traffic_file)
        
        # 2. 为正常数据添加 'normal' 标签
        print("为正常流量数据添加 'normal' 标签...")
        normal_df['label'] = 'normal'
        
        # 3. 加载已标记的异常数据
        print(f"正在加载异常数据: {labeled_anomalies_file}")
        anomalies_df = pd.read_csv(labeled_anomalies_file)
        
        # 4. 合并两个数据集
        print("正在合并数据集...")
        combined_df = pd.concat([normal_df, anomalies_df], ignore_index=True)
        
        # 5. 保存合并后的数据集
        print(f"正在将合并后的数据保存到: {output_file}")
        combined_df.to_csv(output_file, index=False)
        
        print("\n数据集准备完成！")
        print(f"总计 {len(combined_df)} 条数据已保存。")
        print(f"正常样本数: {len(normal_df)}")
        print(f"异常样本数: {len(anomalies_df)}")

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    prepare_data() 