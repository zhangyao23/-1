import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- 全局路径和参数定义 ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
NORMAL_TRAFFIC_FILE = os.path.join(DATA_DIR, 'normal_traffic.csv')
LABELED_ANOMALIES_FILE = os.path.join(DATA_DIR, 'labeled_anomalies.csv')
RANDOM_STATE = 42

def load_full_dataset():
    """加载完整的、带真实标签的数据集"""
    print("--- 步骤 1: 加载完整数据集 ---")
    try:
        df_normal = pd.read_csv(NORMAL_TRAFFIC_FILE)
        df_normal['true_label'] = 0 # 0 for normal
        
        df_anomaly = pd.read_csv(LABELED_ANOMALIES_FILE)
        df_anomaly['true_label'] = 1 # 1 for anomaly
        # 在无监督流程中，我们不需要原始的精细标签
        df_anomaly = df_anomaly.drop('label', axis=1)

        df_full = pd.concat([df_normal, df_anomaly], ignore_index=True)
        print(f"完整数据集加载完毕，共 {len(df_full)} 条样本。")
        return df_full
    except FileNotFoundError:
        print("错误: 数据文件未找到。请先运行 'scripts/generate_initial_data.py'")
        return None

def main():
    """
    执行完整的、修正了评估流程的V1.0工作流
    """
    # 1. 加载数据
    df_full = load_full_dataset()
    if df_full is None:
        return

    # 2. 从源头进行训练/验证集划分 (关键修复)
    print("\n--- 步骤 2: 从源头划分训练集与最终验证集 ---")
    X = df_full.drop('true_label', axis=1)
    y = df_full['true_label']
    
    # 划分70%作为模拟真实世界流程的"训练池"，30%作为完全独立的"最终验证集"
    X_pool, X_final_test, y_pool, y_final_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"数据已划分为训练池({len(X_pool)}条)和最终验证集({len(X_final_test)}条)。")
    print("接下来的所有探索、标注、训练都只在'训练池'中进行。")

    # --- V1.0 流程开始 (仅在训练池上操作) ---

    # 3. 无监督探索 (在训练池上)
    print("\n--- 步骤 3: 在训练池上进行无监督探索 ---")
    scaler = StandardScaler()
    X_pool_scaled = scaler.fit_transform(X_pool)
    
    # 使用与之前相同的参数，看其在更复杂数据上的表现
    dbscan = DBSCAN(eps=5.0, min_samples=5)
    pool_clusters = dbscan.fit_predict(X_pool_scaled)
    X_pool['cluster'] = pool_clusters
    
    # 4. 模拟人工标注 (从训练池的聚类结果中采样)
    print("\n--- 步骤 4: 从训练池的聚类结果中模拟人工标注 ---")
    core_clusters = X_pool[X_pool['cluster'] != -1]
    if core_clusters.empty:
        print("错误: DBSCAN未能发现任何核心簇。流程中止。")
        return
        
    normal_cluster_id = core_clusters['cluster'].value_counts().idxmax()
    potential_anomalies = X_pool[X_pool['cluster'] != normal_cluster_id]
    
    num_to_sample = min(50, len(potential_anomalies))
    if num_to_sample == 0:
        print("错误: 未发现可供标注的潜在异常样本。流程中止。")
        return

    labeled_anomalies = potential_anomalies.sample(n=num_to_sample, random_state=RANDOM_STATE)
    labeled_anomalies['label'] = 1 # 标注为异常

    # 5. 准备最终的监督学习训练集
    print("\n--- 步骤 5: 准备监督学习的训练集 ---")
    normal_samples_for_training = X_pool[X_pool['cluster'] == normal_cluster_id].sample(
        n=num_to_sample * 10, random_state=RANDOM_STATE
    )
    normal_samples_for_training['label'] = 0 # 标注为正常
    
    # 合并成最终的训练集
    df_supervised_train = pd.concat([
        labeled_anomalies.drop('cluster', axis=1), 
        normal_samples_for_training.drop('cluster', axis=1)
    ])
    
    X_supervised_train = df_supervised_train.drop('label', axis=1)
    y_supervised_train = df_supervised_train['label']
    print(f"监督学习训练集准备完毕，共 {len(df_supervised_train)} 条样本。")
    
    # 6. 训练监督模型
    print("\n--- 步骤 6: 训练逻辑回归模型 ---")
    model = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
    # 注意：这里的scaler是用训练池的全部数据fit的，对于监督模型来说，
    # 严谨的做法是重新fit一个只基于`X_supervised_train`的scaler，但为简化PoC，我们暂用旧的。
    # 我们只对要训练和预测的数据进行transform
    X_supervised_train_scaled = scaler.transform(X_supervised_train)
    model.fit(X_supervised_train_scaled, y_supervised_train)
    print("模型训练完成。")
    
    # 7. 在最终验证集上进行公正评估
    print("\n--- 步骤 7: 在独立的最终验证集上进行公正评估 ---")
    X_final_test_scaled = scaler.transform(X_final_test)
    y_final_pred = model.predict(X_final_test_scaled)
    
    print("\n最终模型性能评估:")
    print("\n混淆矩阵:")
    print(confusion_matrix(y_final_test, y_final_pred))
    
    print("\n分类报告 (Classification Report):")
    report = classification_report(y_final_test, y_final_pred, target_names=['Normal (0)', 'Anomaly (1)'])
    print(report)
    
    print("\n结论: 这是V1.0流程在'现实世界'数据集上、无数据泄露情况下的真实性能。")

if __name__ == "__main__":
    main() 