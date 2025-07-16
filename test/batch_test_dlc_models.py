#!/usr/bin/env python3
"""
批量测试当前目录下所有DLC模型文件，输出每个模型的6种异常类型推理结果。
"""
import os
import subprocess

def batch_test_dlc_models():
    # 查找所有.dlc模型文件
    dlc_files = [f for f in os.listdir('.') if f.endswith('.dlc')]
    if not dlc_files:
        print("未找到任何.dlc模型文件！")
        return
    print(f"共找到{len(dlc_files)}个DLC模型文件：{dlc_files}")
    print("\n开始批量测试...\n")
    for dlc_file in dlc_files:
        print("="*60)
        print(f"正在测试模型: {dlc_file}")
        print("="*60)
        # 备份原始模型
        if os.path.exists('multitask_model.dlc'):
            os.rename('multitask_model.dlc', 'multitask_model.dlc.bak')
        # 复制当前模型为 multitask_model.dlc
        subprocess.run(['cp', dlc_file, 'multitask_model.dlc'])
        # 调用测试脚本
        result = subprocess.run(['python3', 'test_dlc_openwrt.py'], capture_output=True, text=True)
        # 输出关键信息
        lines = result.stdout.splitlines()
        print("\n--- 6种异常类型推理结果 ---")
        output_flag = False
        for line in lines:
            if '===== 6种异常类型推理结果 =====' in line:
                output_flag = True
            if output_flag:
                print(line)
            if output_flag and '测试结果汇总' in line:
                break
        print("\n")
        # 恢复原始模型
        if os.path.exists('multitask_model.dlc.bak'):
            os.rename('multitask_model.dlc.bak', 'multitask_model.dlc')

if __name__ == "__main__":
    batch_test_dlc_models() 