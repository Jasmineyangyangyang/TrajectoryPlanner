import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. 极其丝滑地加载 .npz 数据集
dataset = np.load('trajectory_clustering_dataset.npz')
s_coords = dataset['s_coordinates']  # 提取 x 轴
X_features = dataset['d_matrix']     # 提取聚类特征矩阵 X (Shape: N x M)

print(f"成功加载数据集！包含 {X_features.shape[0]} 条轨迹，每条轨迹 {X_features.shape[1]} 个空间特征点。")

# 2. 直接喂给 Scikit-Learn 聚类算法！
# 假设我们想聚成 3 类 (保守、居中、激进切弯)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_features)  # 瞬间完成聚类！

# 获取聚类中心 (也就是每类人群的平均“代表性基准轨迹”)
cluster_centers = kmeans.cluster_centers_ 

# 3. 简单的可视化验证
plt.figure(figsize=(10, 5))
colors = ['r', 'b', 'g']
for i in range(3):
    # 画出每一类的代表性偏好轨迹
    plt.plot(s_coords, cluster_centers[i], color=colors[i], linewidth=3, label=f'Cluster {i+1} Center')

plt.xlabel('纵向弧长 s [m]')
plt.ylabel('横向偏移 d(s) [m]')
plt.title('K-Means 聚类提取的驾驶员典型切弯偏好')
plt.legend()
plt.grid(True)
plt.show()