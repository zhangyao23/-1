import os

# 定义文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
NORMAL_TRAFFIC_FILE = os.path.join(DATA_DIR, 'normal_traffic.csv')

def fix_broken_csv_file():
    """
    修复因缺少换行符而损坏的CSV文件。
    """
    print(f"正在尝试修复文件: {NORMAL_TRAFFIC_FILE}")

    try:
        with open(NORMAL_TRAFFIC_FILE, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        found_broken_line = False
        
        for i, line in enumerate(lines):
            # 默认每行应该有28个字段，逗号分隔就是27个
            # 损坏的行会有更多字段
            if line.count(',') > 30: # 使用一个比27大的阈值来识别
                print(f"发现损坏的行在行号 {i+1}。正在尝试修复...")
                found_broken_line = True
                
                # 我们知道它是由两行各有28个字段的数据组成的
                parts = line.strip().split(',')
                if len(parts) > 28: # 确保有足够的部分可以拆分
                    line1_parts = parts[:28]
                    line2_parts = parts[28:]
                    
                    # 重新组合成两行
                    fixed_lines.append(','.join(line1_parts) + '\\n')
                    fixed_lines.append(','.join(line2_parts) + '\\n')
                    print("成功将行拆分为两行。")
                else:
                    # 如果字段数不符合预期，保留原始行并发出警告
                    print(f"警告: 行 {i+1} 字段数为 {len(parts)}，无法按预期修复。")
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        if found_broken_line:
            print("正在将修复后的内容写回文件...")
            # 将修复后的内容写回原文件
            with open(NORMAL_TRAFFIC_FILE, 'w') as f:
                f.writelines(fixed_lines)
            print("文件修复成功。")
        else:
            print("没有发现需要修复的行。")

    except Exception as e:
        print(f"修复过程中发生错误: {e}")

if __name__ == "__main__":
    fix_broken_csv_file() 