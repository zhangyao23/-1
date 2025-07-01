#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
标签检查脚本

本脚本用于检查 `labeled_anomalies.csv` 文件中所有唯一的标签值。
"""

import pandas as pd
from pathlib import Path

def check_labels():
    """
    读取CSV文件并打印所有唯一的标签。
    """
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data' / 'labeled_anomalies.csv'
    
    print(f"正在分析文件: {data_file}...")
    
    try:
        df = pd.read_csv(data_file)
        
        if 'label' not in df.columns:
            print("错误: 文件中未找到 'label' 列。")
            return
            
        unique_labels = df['label'].unique()
        
        print("\n文件中包含的唯一标签如下:")
        for label in unique_labels:
            print(f"- {label}")
            
    except FileNotFoundError:
        print(f"错误: 文件 '{data_file}' 未找到。")
    except Exception as e:
        print(f"分析过程中发生错误: {e}")

if __name__ == "__main__":
    check_labels() 