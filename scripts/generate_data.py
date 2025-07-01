import os
import csv
import random
import numpy as np

# 定义数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
NORMAL_TRAFFIC_FILE = os.path.join(DATA_DIR, 'normal_traffic.csv')
LABELED_ANOMALIES_FILE = os.path.join(DATA_DIR, 'labeled_anomalies.csv')

# 定义要生成的数据量
NUM_NORMAL_SAMPLES = 2000
NUM_ANOMALY_SAMPLES_PER_TYPE = 200

def generate_normal_traffic():
    """
    生成更多的正常流量数据
    """
    print(f"正在生成 {NUM_NORMAL_SAMPLES} 条新的正常流量数据...")
    with open(NORMAL_TRAFFIC_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        for _ in range(NUM_NORMAL_SAMPLES):
            # 正常流量数据是28个在[-1, 1]范围内的浮点数
            row = np.random.uniform(-1, 1, 28).tolist()
            writer.writerow(row)
    print(f"成功将 {NUM_NORMAL_SAMPLES} 条数据追加到 {NORMAL_TRAFFIC_FILE}")

def generate_anomalies():
    """
    生成更多的异常流量数据
    """
    if not os.path.exists(LABELED_ANOMALIES_FILE):
        print(f"错误: {LABELED_ANOMALIES_FILE} 不存在。")
        return

    # 读取现有的异常数据以学习模式
    with open(LABELED_ANOMALIES_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        existing_anomalies = [row for row in reader]

    anomaly_patterns = {}
    for row in existing_anomalies:
        label = row[-1]
        if label not in anomaly_patterns:
            anomaly_patterns[label] = []
        features = [float(x) for x in row[:-1]]
        anomaly_patterns[label].append(features)

    print(f"正在为每种异常类型生成 {NUM_ANOMALY_SAMPLES_PER_TYPE} 条新的异常数据...")
    new_anomalies = []
    for label, patterns in anomaly_patterns.items():
        for _ in range(NUM_ANOMALY_SAMPLES_PER_TYPE):
            # 从现有模式中随机选择一个作为模板
            base_pattern = random.choice(patterns)
            new_row = base_pattern.copy()
            # 在模板基础上增加一些噪声
            noise = np.random.normal(0, 0.1, len(new_row))
            new_row = [x + n for x, n in zip(new_row, noise)]
            new_row.append(label)
            new_anomalies.append(new_row)

    with open(LABELED_ANOMALIES_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_anomalies)
    
    total_new = len(new_anomalies)
    print(f"成功将 {total_new} 条数据追加到 {LABELED_ANOMALIES_FILE}")


if __name__ == "__main__":
    generate_normal_traffic()
    generate_anomalies()
    print("数据扩充完成。") 