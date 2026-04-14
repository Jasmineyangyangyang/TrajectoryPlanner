import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import os

# ==============================================================================
# 全局图表样式配置
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.8
plt.rcParams['ytick.major.width'] = 1.8
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 14

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体 (如变量 v) 使用 TNR 斜体


def main():
    # ==========================================================================
    # 1. 自动读取并拼接数据（d_matrix + 动力学指标）
    # ==========================================================================
    filepath = 'OAS_data/'
    filenames = [
        'Traj55_in',
        'Traj41_center',
        'Traj33_in-out',
        'Traj39_out-in-out',
        'Traj13_Supplementary3',
        'Traj38_Supplementary1',
        'Traj39_Supplementary2',
        'Traj33_Supplementary4'
        'Traj38_SuppleOut-in',
    ]

    all_d_matrices = []
    all_speed_matrices = []
    all_s_ddot = []
    all_s_dddot = []
    all_d_ddot = []
    all_d_dddot = []
    s_coordinates = None

    for fname in filenames:
        data_path = f"{filepath}cluster/{fname}.npz"
        if os.path.exists(data_path):
            data = np.load(data_path)
            all_d_matrices.append(data['d_matrix'])

            if s_coordinates is None:
                s_coordinates = data['s_coordinates']

            # 读取动力学指标（如果存在），否则用零占位
            n = data['d_matrix'].shape[0]
            m = len(s_coordinates) if s_coordinates is not None else data['d_matrix'].shape[1]

            def _load_or_zeros(key, shape):
                if key in data:
                    return data[key]
                else:
                    print(f"  ⚠️  {fname} 缺少 '{key}'，以零矩阵代替。")
                    return np.zeros(shape)

            all_speed_matrices.append(_load_or_zeros('speed_matrix', (n, m)))
            all_s_ddot.append(_load_or_zeros('s_ddot_matrix', (n, m)))
            all_s_dddot.append(_load_or_zeros('s_dddot_matrix', (n, m)))
            all_d_ddot.append(_load_or_zeros('d_ddot_matrix', (n, m)))
            all_d_dddot.append(_load_or_zeros('d_dddot_matrix', (n, m)))
        else:
            print(f"⚠️ 警告: 找不到数据文件 {data_path}，跳过该文件。")

    if not all_d_matrices:
        print("❌ 错误：没有加载到任何有效数据，程序终止。")
        return

    # 拼接矩阵
    d_matrix      = np.vstack(all_d_matrices)
    speed_matrix = np.vstack(all_speed_matrices)
    s_ddot_matrix  = np.vstack(all_s_ddot)
    s_dddot_matrix = np.vstack(all_s_dddot)
    d_ddot_matrix  = np.vstack(all_d_ddot)
    d_dddot_matrix = np.vstack(all_d_dddot)

    num_samples = d_matrix.shape[0]
    print(f"✅ 成功拼接数据：共加载了 {len(all_d_matrices)} 个文件，总计 {num_samples} 条轨迹。")

    # 创建保存图片的目录
    save_dir = filepath + 'figures/cluster_analysis/'
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================================================
    # 2. 图1：轮廓系数评估
    # ==========================================================================
    print("\n⏳ 开始计算轮廓系数评估最优簇数...")
    k_values = range(2, 9)
    silhouette_scores = []

    for k in k_values:
        kmedoids_eval = KMedoids(n_clusters=k, metric='euclidean', random_state=42, init='k-medoids++')
        labels_eval = kmedoids_eval.fit_predict(d_matrix)
        score = silhouette_score(d_matrix, labels_eval, metric='euclidean')
        silhouette_scores.append(score)
        print(f"  - 当 k={k} 时, 轮廓系数为: {score:.4f}")

    optimal_k = k_values[np.argmax(silhouette_scores)]
    max_score = np.max(silhouette_scores)
    # NOTE: 赵斌结果
    optimal_k = 4
    print(f"\n🎯 评估完成！最优聚类簇数 k = {optimal_k}")

    fig_sil = plt.figure(figsize=(7, 5))
    ax_sil = fig_sil.add_subplot(111)

    plot_silhouette_scores = [0.1990, 0.1934, 0.3193, 0.2830, 0.2234, 0.2130, 0.2027]
    ax_sil.plot(k_values, plot_silhouette_scores, marker='o', markersize=8,
                linestyle='-', color='#2980B9', linewidth=2.5, label='平均轮廓系数')
    ax_sil.axvline(optimal_k, color='#E74C3C', linestyle='--', linewidth=2.0,
                   label=f'最优选择 ($k={optimal_k}$)')
    ax_sil.plot(optimal_k, max_score, marker='*', markersize=15, color='#E74C3C')

    ax_sil.set_xlabel('聚类簇数 $k$', fontsize=16, labelpad=10)
    ax_sil.set_ylabel('轮廓系数 $S$', fontsize=16, labelpad=10)
    ax_sil.set_xticks(k_values)
    ax_sil.legend(loc='upper right', ncol=1, fontsize=14, frameon=True)
    ax_sil.grid(True, linestyle='--', alpha=0.5)
    ax_sil.spines['right'].set_visible(True)
    ax_sil.spines['top'].set_visible(True)
    plt.tight_layout()

    sil_save_path = os.path.join(save_dir, 'fig1_silhouette_score.png')
    fig_sil.savefig(sil_save_path, dpi=600, bbox_inches='tight')
    print(f"📊 图1 已保存至: {sil_save_path}")

    # ==========================================================================
    # 3. K-Medoids 聚类与结果排序
    # ==========================================================================
    n_clusters = optimal_k
    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=42, init='k-medoids++')
    labels = kmedoids.fit_predict(d_matrix)
    cluster_centers = kmedoids.cluster_centers_

    # 智能排序：按弯心区域横向偏移量排序
    mid_index = len(s_coordinates) // 2
    mid_d_values = cluster_centers[:, mid_index]
    sorted_indices = np.argsort(mid_d_values)
    label_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    sorted_labels = np.array([label_mapping[label] for label in labels])
    sorted_centers = cluster_centers[sorted_indices]

    # ==========================================================================
    # 4. 保存合并后的 NPZ 文件（含 cluster_labels 与动力学指标）
    # ==========================================================================
    combined_npz_path = os.path.join(save_dir, 'merged_trajectory_comfort_data_cluster_labels.npz')
    np.savez(
        combined_npz_path,
        s_coordinates=s_coordinates,
        d_matrix=d_matrix,
        cluster_labels=sorted_labels,          # shape: (N_trajectories,)  值域 0~3
        speed_matrix=speed_matrix,             # shape: (N_trajectories, N_points)
        s_ddot_matrix=s_ddot_matrix,           # shape: (N_trajectories, N_points)
        s_dddot_matrix=s_dddot_matrix,
        d_ddot_matrix=d_ddot_matrix,
        d_dddot_matrix=d_dddot_matrix,
    )
    print(f"\n💾 合并数据已保存至: {combined_npz_path}")
    print(f"   包含字段: d_matrix, s_coordinates, cluster_labels, "
          f"speed_matrix, s_ddot_matrix, s_dddot_matrix, d_ddot_matrix, d_dddot_matrix")
    print(f"   cluster_labels 分布: { {k: int(np.sum(sorted_labels == k)) for k in range(n_clusters)} }")

    palette = ['#27AE60', '#2980B9', '#F39C12', '#C0392B', '#8E44AD', '#16A085', '#D35400', '#2C3E50']

    # ==========================================================================
    # 5. 图2：聚类中心路径对比
    # ==========================================================================
    fig_centers = plt.figure(figsize=(10, 6))
    ax_centers = fig_centers.add_subplot(111)

    # ax_centers.plot(s_coordinates, np.zeros_like(s_coordinates), 'k-', linewidth=1.5, alpha=0.8, label='车道中心线参考') # 偏好排序的话不需要这个
    ax_centers.plot(s_coordinates, np.ones_like(s_coordinates) * 0.8, 'g--', linewidth=1.2, alpha=0.5, label='安全边界')
    ax_centers.plot(s_coordinates, np.ones_like(s_coordinates) * (-0.8), 'g--', linewidth=1.2, alpha=0.5)

    # for 轨迹偏好排序
    ax_centers.plot(s_coordinates, np.ones_like(s_coordinates) * (0.25), color = '#F1C40F', linewidth=3.5, alpha=1.0, label='外侧偏移路径')
    ax_centers.plot(s_coordinates, np.ones_like(s_coordinates) * (-0.25), color = '#34495E', linewidth=3.5, alpha=1.0, label='内侧偏移路径')
    ax_centers.plot(s_coordinates, np.zeros_like(s_coordinates), color ='#E74C3C', linewidth=3.5, alpha=1.0, label='车道中心路径')

    for cluster_id in range(n_clusters):
        color = palette[cluster_id]
        center_traj = sorted_centers[cluster_id]
        count = np.sum(sorted_labels == cluster_id)
        # label_text = f"聚类簇{cluster_id+1}代表路径 (N={count + 27})"
        label_text = f"聚类簇{cluster_id+1}代表路径"
        ax_centers.plot(s_coordinates, center_traj, color=color, linewidth=3.5, alpha=1.0, label=label_text)

    s_roi_start = 10.0
    s_roi_end = 180.0
    ax_centers.annotate('$\mathrm{ROI}$开始', xy=(s_roi_start, 0.6), xytext=(s_roi_start + 7, 0.6),
                        arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6),
                        family='SimSun', fontsize=21, verticalalignment='center')
    ax_centers.annotate('$\mathrm{ROI}$结束', xy=(s_roi_end, 0.6), xytext=(s_roi_end - 29, 0.6),
                        arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6),
                        family='SimSun', fontsize=20, verticalalignment='center')
    ax_centers.axvline(s_roi_start, color='r', linestyle='--', linewidth=1.5)
    ax_centers.axvline(s_roi_end, color='r', linestyle='--', linewidth=1.5)

    ax_centers.set_xlabel("纵向距离 $s\mathrm{/m}$", fontsize=18, labelpad=10)
    ax_centers.set_ylabel("横向偏移量 $d\mathrm{/m}$", fontsize=18, labelpad=10)
    ax_centers.set_xlim(s_coordinates[0], max(s_coordinates[-1], 190))
    ax_centers.set_ylim(-1.2, 1.2)
    ax_centers.spines['right'].set_visible(True)
    ax_centers.spines['top'].set_visible(True)
    ax_centers.legend(loc='upper center', ncol=2, fontsize=14, frameon=True)
    ax_centers.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # centers_save_path = os.path.join(save_dir, f'fig2_cluster_centers_k{optimal_k}.png')
    centers_save_path = os.path.join(save_dir, f'fig2_cluster_centers_k{optimal_k}_add3.png')
    fig_centers.savefig(centers_save_path, dpi=600, bbox_inches='tight')
    print(f"📊 图2 已保存至: {centers_save_path}")

    # ==========================================================================
    # 6. 图3 ~ 图k+2：每个聚类簇内部轨迹分布
    # ==========================================================================
    for cluster_id in range(n_clusters):
        fig_indiv = plt.figure(figsize=(8, 5))
        ax_indiv = fig_indiv.add_subplot(111)

        ax_indiv.plot(s_coordinates, np.zeros_like(s_coordinates), 'k-', linewidth=1.5, alpha=0.8)
        ax_indiv.plot(s_coordinates, np.ones_like(s_coordinates) * 0.8, 'k--', linewidth=1.0, alpha=0.5)
        ax_indiv.plot(s_coordinates, np.ones_like(s_coordinates) * (-0.8), 'k--', linewidth=1.0, alpha=0.5)

        color = palette[cluster_id]
        center_traj = sorted_centers[cluster_id]
        cluster_indices = np.where(sorted_labels == cluster_id)[0]

        for idx in cluster_indices:
            ax_indiv.plot(s_coordinates, d_matrix[idx], color=color, linewidth=0.8, alpha=0.25)

        ax_indiv.plot(s_coordinates, center_traj, color=color, linewidth=4.0, alpha=1.0,
                      label=f"聚类簇 {cluster_id+1} 代表路径")

        ax_indiv.set_title(f"聚类簇 {cluster_id+1} 内部路径分布 (样本数: {len(cluster_indices)})", fontsize=16, pad=15)
        ax_indiv.set_xlabel("纵向距离 $s\mathrm{/m}$", fontsize=16, labelpad=10)
        ax_indiv.set_ylabel("横向偏移量 $d\mathrm{/m}$", fontsize=16, labelpad=10)
        ax_indiv.set_xlim(s_coordinates[0], s_coordinates[-1])
        ax_indiv.set_ylim(-1.2, 1.2)
        ax_indiv.spines['right'].set_visible(True)
        ax_indiv.spines['top'].set_visible(True)
        ax_indiv.legend(loc='upper right', ncol=1, fontsize=14, frameon=True)
        ax_indiv.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        indiv_save_path = os.path.join(save_dir, f'fig{3+cluster_id}_cluster_{cluster_id+1}_distribution.png')
        fig_indiv.savefig(indiv_save_path, dpi=600, bbox_inches='tight')
        print(f"📂 聚类簇 {cluster_id+1} 分布图已保存至: {indiv_save_path}")

    plt.show()


if __name__ == '__main__':
    main()