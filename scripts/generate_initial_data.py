import os
import csv
import numpy as np
import random
from sklearn.datasets import make_blobs

# 定义数据目录和文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
NORMAL_TRAFFIC_FILE = os.path.join(DATA_DIR, 'normal_traffic.csv')
LABELED_ANOMALIES_FILE = os.path.join(DATA_DIR, 'labeled_anomalies.csv')

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 定义要生成的数据量 - 更符合现实的分布
NUM_NORMAL_SAMPLES = 18000  # 正常数据占90%
NUM_FEATURES = 6
NUM_ANOMALY_CLUSTERS = 7
SAMPLES_PER_CLUSTER = 200   # 每个异常类别200个样本
NUM_BOUNDARY_SAMPLES = 150  # 边界模糊的难分类点

def generate_realistic_normal_traffic():
    """
    生成90%聚集的正常流量数据，模拟真实场景中大部分数据的集中分布
    """
    print(f"正在生成 {NUM_NORMAL_SAMPLES} 条现实场景的正常流量数据...")
    
    # 主要正常数据集群 (占80%的正常数据)
    main_cluster_size = int(NUM_NORMAL_SAMPLES * 0.8)
    main_center = np.zeros(NUM_FEATURES)
    main_cluster = np.random.multivariate_normal(
        mean=main_center,
        cov=np.eye(NUM_FEATURES) * 0.5,  # 较小的方差，形成紧密集群
        size=main_cluster_size
    )
    
    # 次要正常数据散布 (占20%的正常数据)
    scatter_size = NUM_NORMAL_SAMPLES - main_cluster_size
    scatter_data = np.random.multivariate_normal(
        mean=main_center,
        cov=np.eye(NUM_FEATURES) * 1.2,  # 较大的方差，形成外围散布
        size=scatter_size
    )
    
    # 合并正常数据
    normal_data = np.vstack([main_cluster, scatter_data])
    np.random.shuffle(normal_data)
    
    with open(NORMAL_TRAFFIC_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'feature_{i}' for i in range(NUM_FEATURES)]
        writer.writerow(header)
        writer.writerows(normal_data)
    
    print(f"成功创建 {NORMAL_TRAFFIC_FILE}")
    return main_center

def generate_realistic_anomalies(normal_center):
    """
    生成符合现实场景的异常数据：
    1. 7个明确的异常类别集群
    2. 少量边界模糊的难分类点
    """
    print("正在生成现实场景的异常数据...")
    
    all_anomaly_data = []
    
    # 为每个异常集群定义明确的中心点，与正常数据保持足够距离
    cluster_centers = []
    
    # 生成7个异常集群的中心点
    for i in range(NUM_ANOMALY_CLUSTERS):
        # 在不同的方向和距离创建异常中心
        angle = (2 * np.pi * i) / NUM_ANOMALY_CLUSTERS  # 均匀分布角度
        distance = np.random.uniform(4, 6)  # 与正常数据的距离
        
        center = normal_center.copy()
        # 在几个关键维度上创建明显偏移
        key_dims = np.random.choice(NUM_FEATURES, size=np.random.randint(3, 6), replace=False)
        
        for dim in key_dims:
            if dim < 2:  # 在前两个维度使用角度分布
                center[dim] += distance * np.cos(angle) if dim == 0 else distance * np.sin(angle)
            else:
                center[dim] += np.random.uniform(-distance, distance)
        
        cluster_centers.append(center)
    
    # 生成每个异常集群的数据
    for i, center in enumerate(cluster_centers):
        cluster_label = f'anomaly_cluster_{i+1}'
        
        # 生成集群数据 - 有一定内部变异但整体聚集
        cluster_data = np.random.multivariate_normal(
            mean=center,
            cov=np.eye(NUM_FEATURES) * 0.8,  # 适中的方差，形成明确但不完美的集群
            size=SAMPLES_PER_CLUSTER
        )
        
        # 为每个数据点添加标签
        for data_point in cluster_data:
            all_anomaly_data.append(list(data_point) + [cluster_label])
    
    # 生成边界模糊的难分类点
    print(f"正在生成 {NUM_BOUNDARY_SAMPLES} 个边界模糊的难分类点...")
    
    for _ in range(NUM_BOUNDARY_SAMPLES):
        # 随机选择两个集群之间的区域
        cluster_a, cluster_b = np.random.choice(len(cluster_centers), size=2, replace=False)
        center_a, center_b = cluster_centers[cluster_a], cluster_centers[cluster_b]
        
        # 在两个集群中心之间生成点
        interpolation_factor = np.random.uniform(0.3, 0.7)  # 避免完全在中心
        boundary_center = interpolation_factor * center_a + (1 - interpolation_factor) * center_b
        
        # 添加额外的随机性
        noise = np.random.multivariate_normal(
            mean=np.zeros(NUM_FEATURES),
            cov=np.eye(NUM_FEATURES) * 1.0,
            size=1
        )[0]
        
        boundary_point = boundary_center + noise
        
        # 随机分配给其中一个类别（模拟标注的主观性/困难性）
        assigned_label = f'anomaly_cluster_{np.random.choice([cluster_a, cluster_b]) + 1}'
        all_anomaly_data.append(list(boundary_point) + [assigned_label])
    
    # 打乱数据顺序
    np.random.shuffle(all_anomaly_data)
    
    # 保存到文件
    with open(LABELED_ANOMALIES_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'feature_{i}' for i in range(NUM_FEATURES)] + ['label']
        writer.writerow(header)
        writer.writerows(all_anomaly_data)
    
    total_anomalies = len(all_anomaly_data)
    print(f"成功创建 {LABELED_ANOMALIES_FILE}，包含 {total_anomalies} 条异常数据")
    print(f"  - {NUM_ANOMALY_CLUSTERS} 个明确异常集群，每个 {SAMPLES_PER_CLUSTER} 样本")
    print(f"  - {NUM_BOUNDARY_SAMPLES} 个边界模糊的难分类点")

def generate_data_distribution_summary():
    """
    输出数据分布摘要，帮助理解生成的数据特征
    """
    print("\n=== 数据分布摘要 ===")
    print(f"正常数据: {NUM_NORMAL_SAMPLES} 条 (约90%聚集分布)")
    print(f"异常数据: {NUM_ANOMALY_CLUSTERS * SAMPLES_PER_CLUSTER + NUM_BOUNDARY_SAMPLES} 条")
    print(f"  - 明确异常集群: {NUM_ANOMALY_CLUSTERS} 个 × {SAMPLES_PER_CLUSTER} 样本")
    print(f"  - 边界模糊点: {NUM_BOUNDARY_SAMPLES} 个")
    print(f"特征维度: {NUM_FEATURES}")
    print("===================\n")

if __name__ == "__main__":
    print("开始生成符合现实场景的训练数据...")
    generate_data_distribution_summary()
    
    # 生成正常数据并获取中心点
    normal_center = generate_realistic_normal_traffic()
    
    # 基于正常数据中心生成异常数据
    generate_realistic_anomalies(normal_center)
    
    print("现实场景数据生成完成！")
    print("这个数据集特点：")
    print("- 正常数据90%聚集，有自然的分散性")
    print("- 异常类别形成明确但不完美的集群")
    print("- 包含边界模糊的难分类样本")
    print("- 更接近真实网络异常检测场景") 