import os
os.environ["OMP_NUM_THREADS"] = "1"  # 禁用OpenMP多线程
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载MKL库

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成示例数据（确保数据是浮点型）
X, _ = make_blobs(n_samples=300, centers=3, random_state=42, dtype=np.float64)

# 显式设置K-Means参数避免警告
kmeans = KMeans(
    n_clusters=3,
    n_init=10,       # 显式指定初始化次数
    init='k-means++', # 使用更稳定的初始化方式
    random_state=42
).fit(X)

# 获取结果
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 计算类内距离（增加异常处理）
intra_distances = []
for i in range(kmeans.n_clusters):
    try:
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:  # 确保簇不为空
            distances = np.linalg.norm(cluster_points - centers[i], axis=1)
            intra_distances.append(np.mean(distances))
        else:
            intra_distances.append(0)  # 空簇距离设为0
    except Exception as e:
        print(f"计算簇 {i} 距离时出错: {str(e)}")
        intra_distances.append(np.nan)  # 标记错误

# 排序并输出
sorted_clusters = sorted(zip(range(kmeans.n_clusters), intra_distances),
                        key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))

print("簇排序结果（从紧凑到分散）:")
for cluster, distance in sorted_clusters:
    print(f"簇 {cluster}: 平均距离 = {distance:.4f}")

# 可视化（增加图形后端设置）
plt.switch_backend('TkAgg')  # 使用更稳定的图形后端
colors = plt.cm.rainbow(np.linspace(0, 1, kmeans.n_clusters))
for (cluster, _), color in zip(sorted_clusters, colors):
    mask = labels == cluster
    if np.any(mask):  # 确保有数据点
        plt.scatter(X[mask, 0], X[mask, 1], color=color,
                   label=f'簇 {cluster} (距离={intra_distances[cluster]:.2f})')

plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='black')
plt.legend()
plt.title("K-Means聚类结果（按类内距离排序）")
plt.show()
