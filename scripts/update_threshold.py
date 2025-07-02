#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
更新自编码器模型的异常检测阈值

用于手动调整已训练模型的检测敏感度
"""

import os
import sys
import json
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger.system_logger import SystemLogger
from src.ai_models.autoencoder_model import AutoencoderModel

def update_threshold(new_threshold: float):
    """更新自编码器模型的阈值"""
    try:
        # 加载配置
        config_path = os.path.join(project_root, 'config', 'system_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 初始化日志
        logger = SystemLogger(config)
        logger.setup_logger("ThresholdUpdater")
        
        autoencoder_config = config['ai_models']['autoencoder']
        
        # 创建自编码器模型实例
        model = AutoencoderModel(autoencoder_config, logger)
        
        # 更新阈值
        old_threshold = model.threshold
        model.update_threshold(new_threshold)
        
        # 保存优化后的阈值
        model.save_optimized_threshold(new_threshold)
        
        print(f"✅ 阈值更新成功！")
        print(f"   旧阈值: {old_threshold:.6f}")
        print(f"   新阈值: {new_threshold:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 阈值更新失败: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='更新自编码器异常检测阈值')
    parser.add_argument('threshold', type=float, help='新的阈值（建议范围: 1.0-5.0）')
    
    args = parser.parse_args()
    
    if args.threshold <= 0:
        print("❌ 阈值必须为正数")
        sys.exit(1)
    
    success = update_threshold(args.threshold)
    sys.exit(0 if success else 1) 