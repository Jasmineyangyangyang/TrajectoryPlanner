import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import os
import matplotlib.patches as mpatches

# ==============================================================================
# 全局图表样式配置 (IEEE 期刊标准)
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 14
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体 (如变量 v) 使用 TNR 斜体


PALETTE = ['#27AE60', '#2980B9', '#F39C12', '#C0392B'] # 绿, 蓝, 黄, 红
CLUSTER_NAMES = ['簇 1', '簇 2', '簇 3', '簇 4']

# ==============================================================================
# 通用绘图与统计辅助函数
# ==============================================================================
def plot_profile(s_coords, data_matrix, labels, ylabel, save_path):
    """绘制均值±标准差的 Profile 曲线图 (全景 [0,190] 强行留白版)"""
    fig, ax = plt.subplots(figsize=(7, 4.2)) 
    
    for cluster_id in range(4):
        cluster_data = data_matrix[labels == cluster_id]
        mean_val = np.mean(cluster_data, axis=0)
        std_val = np.std(cluster_data, axis=0)
        
        ax.plot(s_coords, mean_val, color=PALETTE[cluster_id], linewidth=2.5, 
                label=f'{CLUSTER_NAMES[cluster_id]}')
        ax.fill_between(s_coords, mean_val - std_val, mean_val + std_val, 
                        color=PALETTE[cluster_id], alpha=0.2)

    ax.set_xlabel("纵向距离 $s$$\\mathrm{/m}$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim(0, 190) 
    
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.18)
    
    s_roi_start = 10.0
    s_roi_end = 180.0
    current_ymin, current_ymax = ax.get_ylim()
    anno_y = current_ymin + (current_ymax - current_ymin) * 0.88
    
    ax.axvline(s_roi_start, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(s_roi_end, color='r', linestyle='--', linewidth=1.5)
    
    ax.annotate('$\\mathrm{ROI}$开始', xy=(s_roi_start, anno_y), xytext=(s_roi_start + 10, anno_y),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6),
                family='SimSun', fontsize=18, verticalalignment='center')
        
    ax.annotate('$\\mathrm{ROI}$结束', xy=(s_roi_end, anno_y), xytext=(s_roi_end - 41, anno_y),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6),
                family='SimSun', fontsize=18, verticalalignment='center')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=True, fontsize=12)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


def draw_significance_brackets(ax, data_df, x_col, y_col, posthoc_df):
    """为单列数据图 (Fig5-8) 绘制统计显著性连线"""
    pairs = []
    for i in range(4):
        for j in range(i+1, 4):
            pval = posthoc_df.iloc[i, j]
            if pval < 0.05: 
                pairs.append((i, j, pval))
                
    if not pairs: return

    y_max_base = data_df[y_col].max()
    y_range = y_max_base - data_df[y_col].min()
    current_height = y_max_base + y_range * 0.05
    step = y_range * 0.08
            
    pairs.sort(key=lambda x: abs(x[0] - x[1]))

    for i, j, pval in pairs:
        x1, x2 = i, j
        if pval < 0.001: mark = '***'
        elif pval < 0.01: mark = '**'
        else: mark = '*'

        h = y_range * 0.015
        ax.plot([x1, x1, x2, x2], [current_height, current_height+h, current_height+h, current_height], 
                lw=1.2, color='black')
        ax.text((x1+x2)*0.5, current_height+h, mark, ha='center', va='bottom', 
                color='black', fontsize=12, fontweight='bold')
        current_height += step

    ax.set_ylim(top=current_height + step)


def draw_grouped_significance_brackets(ax, df_phase, metric, table4_df, phases_list):
    """专为分组图 (Fig9-12) 绘制各个 Phase 内的统计显著性连线"""    
    y_max_base = df_phase[metric].max()
    y_range = y_max_base - df_phase[metric].min()
    base_height = y_max_base + y_range * 0.05
    step = y_range * 0.08
    
    # 设定 hue 偏移量 (对于 dodge=True, 4类数据的标准偏移中心点)
    hue_offsets = [-0.3, -0.1, 0.1, 0.3]
    global_max_height = base_height
    
    for p_idx, phase_name in enumerate(phases_list):
        phase_pairs = table4_df[(table4_df['Phase'] == phase_name) & 
                                (table4_df['Indicator'] == metric) & 
                                (table4_df['Adjusted p-value'] < 0.05)]
        if phase_pairs.empty: continue
        
        current_height = base_height
        pairs = []
        for _, row in phase_pairs.iterrows():
            comp = row['Cluster comparison']
            
            # ⭐ 修复点：使用 split 分割字符串，彻底避免下标数错的问题
            parts = comp.split(' vs ')
            c_i = int(parts[0].replace('C', '')) - 1
            c_j = int(parts[1].replace('C', '')) - 1
            
            pval = row['Adjusted p-value']
            pairs.append((c_i, c_j, pval))
            
        pairs.sort(key=lambda x: abs(x[0] - x[1]))
        
        for c_i, c_j, pval in pairs:
            x1 = p_idx + hue_offsets[c_i]
            x2 = p_idx + hue_offsets[c_j]
            
            if pval < 0.001: mark = '***'
            elif pval < 0.01: mark = '**'
            else: mark = '*'
            
            h = y_range * 0.015
            ax.plot([x1, x1, x2, x2], [current_height, current_height+h, current_height+h, current_height], lw=1.0, color='black')
            ax.text((x1+x2)*0.5, current_height+h, mark, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')
            current_height += step
            global_max_height = max(global_max_height, current_height)
            
    ax.set_ylim(top=global_max_height + step)


def main():
    filepath = 'OAS_data/figures/cluster_analysis/'
    save_dir = 'OAS_data/figures/comfort_analysis/'
    os.makedirs(save_dir, exist_ok=True)
    
    data_path = os.path.join(filepath, 'merged_trajectory_comfort_data_cluster_labels.npz')
    data = np.load(data_path)
    s_coords_raw = data['s_coordinates']
    labels = data['cluster_labels']
    
    d_ddot_raw = data['d_ddot_matrix']
    d_dddot_raw = data['d_dddot_matrix']
    s_ddot_raw = data['s_ddot_matrix']
    s_dddot_raw = data['s_dddot_matrix']

    # ==========================================================================
    # PART 1: 整体 AOI 提取与 Fig 1-8 绘制
    # ==========================================================================
    print("✂️ 正在截取有效试验路段 (s ∈ [10, 180])...")
    roi_mask = (s_coords_raw >= 10.0) & (s_coords_raw <= 180.0)
    
    s_coords = s_coords_raw[roi_mask]
    d_ddot = d_ddot_raw[:, roi_mask]
    d_dddot = d_dddot_raw[:, roi_mask]
    s_ddot = s_ddot_raw[:, roi_mask]
    s_dddot = s_dddot_raw[:, roi_mask]

    print("📊 正在绘制 Fig 1-4 (带留白全景的纯净 Profile 图)...")
    plot_profile(s_coords, d_ddot, labels, "横向加速度 $a_{\\mathrm{lat}}$$\\mathrm{\\ /(m/s^2)}$", os.path.join(save_dir, 'Fig1_d_ddot_profile.png'))
    plot_profile(s_coords, d_dddot, labels, "横向加加速度 $j_{\\mathrm{lat}}$$\\mathrm{\\ /(m/s^3)}$", os.path.join(save_dir, 'Fig2_d_dddot_profile.png'))
    plot_profile(s_coords, s_ddot, labels, "纵向加速度 $a_{\\mathrm{lon}}$$\\mathrm{\\ /(m/s^2)}$", os.path.join(save_dir, 'Fig3_s_ddot_profile.png'))
    plot_profile(s_coords, s_dddot, labels, "纵向加加速度 $j_{\\mathrm{lon}}$$\\mathrm{\\ /(m/s^3)}$", os.path.join(save_dir, 'Fig4_s_dddot_profile.png'))

    # 构建整体 ROI 统计指标字典 (添加了准确的带单位 y_label)
    metrics_overall = {
        'max_abs_d_ddot': np.max(np.abs(d_ddot), axis=1),
        'max_abs_d_dddot': np.max(np.abs(d_dddot), axis=1),
        'max_abs_s_ddot': np.max(np.abs(s_ddot), axis=1),
        'max_abs_s_dddot': np.max(np.abs(s_dddot), axis=1),
    }
    df_overall = pd.DataFrame(metrics_overall)
    df_overall['Cluster'] = labels
    
    # ⭐ 完美修复的 Y 轴标签
    metric_ylabels = {
        'max_abs_d_ddot': "最大横向加速度 $a_{\\mathrm{lat,max}}$$\\mathrm{\\ /(m/s^2)}$",
        'max_abs_d_dddot': "最大横向加加速度 $j_{\\mathrm{lat,max}}$$\\mathrm{\\ /(m/s^3)}$",
        'max_abs_s_ddot': "最大纵向加速度 $a_{\\mathrm{lon,max}}$$\\mathrm{\\ /(m/s^2)}$",
        'max_abs_s_dddot': "最大纵向加加速度 $j_{\\mathrm{lon,max}}$$\\mathrm{\\ /(m/s^3)}$"
    }

    print("🎨 正在绘制 Fig 5 - Fig 8 (整体 ROI 分布图)...")
    for idx, metric in enumerate(metrics_overall.keys()):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.violinplot(x='Cluster', y=metric, data=df_overall, ax=ax, palette=PALETTE, 
                       inner=None, linewidth=0, alpha=0.5, hue='Cluster', legend=False)
        sns.boxplot(x='Cluster', y=metric, data=df_overall, ax=ax, width=0.5, 
                    boxprops={'facecolor':'none', 'edgecolor':'black', 'zorder': 10}, 
                    fliersize=0, showcaps=True)
        sns.stripplot(x='Cluster', y=metric, data=df_overall, ax=ax, palette=PALETTE, 
                      size=3, jitter=0.15, alpha=0.7, hue='Cluster', legend=False)

        # 引入 Kruskal-Wallis 检验和 Dunn's post-hoc 检验，并添加显著性标注
        groups_for_kw = [df_overall[df_overall['Cluster'] == i][metric].values for i in range(4)]
        kw_stat, p_val = stats.kruskal(*groups_for_kw)
        if p_val < 0.05:
            # 只有总体验证显著，才提取 posthoc 并画线
            posthoc_df = sp.posthoc_dunn(df_overall, val_col=metric, group_col='Cluster', p_adjust='bonferroni')
            draw_significance_brackets(ax, df_overall, 'Cluster', metric, posthoc_df)
        else:
            print(f"  ⚠️ [提示] {metric} 总体 Kruskal-Wallis 检验不显著 (p={p_val:.4f})，该图将不绘制事后星号。")

        # posthoc_df = sp.posthoc_dunn(df_overall, val_col=metric, group_col='Cluster', p_adjust='bonferroni')
        # draw_significance_brackets(ax, df_overall, 'Cluster', metric, posthoc_df)

        ax.set_xlabel('聚类簇类别', fontsize=14)
        ax.set_ylabel(metric_ylabels[metric], fontsize=14) # 写入带单位的 Y 轴
        ax.set_xticks(range(4))
        ax.set_xticklabels(['簇 1', '簇 2', '簇 3', '簇 4'])
        
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'Fig{5 + idx}_{metric}_Distribution.png'), dpi=600, bbox_inches='tight')
        plt.close()


    # ==========================================================================
    # PART 2: 弯道三阶段切片分析 (Entry, Corner, Exit) 及 Fig 9-12
    # ==========================================================================
    print("\n⏱️ 正在进行弯道多阶段动力学特征提取与分析...")
    phases = {
        # 'Overall': (10.0, 180.0),
        'Entry': (30.0, 70.0),
        'Corner': (70.0, 120.0),
        'Exit': (120.0, 160.0)
    }
    
    phase_metrics_list = []
    
    # 对每条轨迹的每个阶段进行计算
    for phase_name, (s_start, s_end) in phases.items():
        p_mask = (s_coords_raw >= s_start) & (s_coords_raw <= s_end)
        
        d_ddot_p = d_ddot_raw[:, p_mask]
        d_dddot_p = d_dddot_raw[:, p_mask]
        s_ddot_p = s_ddot_raw[:, p_mask]
        s_dddot_p = s_dddot_raw[:, p_mask]
        
        for i in range(len(labels)):
            phase_metrics_list.append({
                'Phase': phase_name,
                'Cluster': labels[i],
                'max_abs_d_ddot': np.max(np.abs(d_ddot_p[i])),
                'max_abs_d_dddot': np.max(np.abs(d_dddot_p[i])), # 此处按要求使用 max
                'max_abs_s_ddot': np.max(np.abs(s_ddot_p[i])),
                'max_abs_s_dddot': np.max(np.abs(s_dddot_p[i]))
            })
            
    df_phase = pd.DataFrame(phase_metrics_list)
    # 固定阶段排序
    # 将 Overall 包含进类别中，锁定排序顺序
    # phases_list_all = ['Overall', 'Entry', 'Corner', 'Exit']
    phases_list_all = ['Entry', 'Corner', 'Exit']
    df_phase['Phase'] = pd.Categorical(df_phase['Phase'], categories=phases_list_all, ordered=True)

    # 阶段统计检验
    phase_metrics_keys = ['max_abs_d_ddot', 'max_abs_d_dddot', 'max_abs_s_ddot', 'max_abs_s_dddot']
    phase_ylabels = {
        'max_abs_d_ddot': "最大横向加速度 $a_{\\mathrm{lat,max}}$$\\mathrm{\\ /(m/s^2)}$",
        'max_abs_d_dddot': "最大横向加加速度 $j_{\\mathrm{lat,max}}$$\\mathrm{\\ /(m/s^3)}$",
        'max_abs_s_ddot': "最大纵向加速度 $a_{\\mathrm{lon,max}}$$\\mathrm{\\ /(m/s^2)}$",
        'max_abs_s_dddot': "最大纵向加加速度 $j_{\\mathrm{lon,max}}$$\\mathrm{\\ /(m/s^3)}$"
    }

    kw_phase_results = []
    dunn_phase_results = []

    for phase_name in phases_list_all:
        df_sub = df_phase[df_phase['Phase'] == phase_name]
        
        for metric in phase_metrics_keys:
            groups = [df_sub[df_sub['Cluster'] == i][metric].values for i in range(4)]
            stat, p = stats.kruskal(*groups) # stat:H 统计量（检验统计值）; p:对应的显著性 p 值
            kw_phase_results.append({'Phase': phase_name, 'Indicator': metric, 'H statistic': round(stat, 3), 'p-value': p})
            
            if p < 0.05:
                posthoc = sp.posthoc_dunn(df_sub, val_col=metric, group_col='Cluster', p_adjust='bonferroni')
                for i in range(4):
                    for j in range(i+1, 4):
                        pval = posthoc.iloc[i, j]
                        z_approx = "N/A" # sp.posthoc_dunn 默认输出为 p 值，若需要Z可自写扩展，这里保留表格结构
                        dunn_phase_results.append({
                            'Phase': phase_name,
                            'Indicator': metric,
                            'Cluster comparison': f'C{i+1} vs C{j+1}',
                            'Z statistic': z_approx, # Z > 0：Group1 的平均秩 > Group2（值更大/更激进）, Z < 0：Group1 的平均秩 < Group2（值更小/更保守）
                            'Adjusted p-value': pval
                        })

    # 保存 Table 1 和 Table 2
    df_kw_phase = pd.DataFrame(kw_phase_results)
    df_dunn_phase = pd.DataFrame(dunn_phase_results)
    df_kw_phase.to_csv(os.path.join(save_dir, 'Table1_Phase_Kruskal_Wallis.csv'), index=False)
    df_dunn_phase.to_csv(os.path.join(save_dir, 'Table2_Phase_Dunn_Posthoc.csv'), index=False)
    print("📝 弯道多阶段统计表格 Table 1 和 Table 2 已导出。")

    # 绘制 Fig 9 - Fig 12
    print("🎨 正在绘制 Fig 9 - Fig 12 (多阶段分组特征分布图)...")
    for idx, metric in enumerate(phase_metrics_keys):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 1. 分组小提琴
        sns.violinplot(x='Phase', y=metric, hue='Cluster', data=df_phase, 
                       palette=PALETTE, inner=None, linewidth=0, alpha=0.5, legend=False, ax=ax)
        # 2. ⭐ 终极修复方案：绕过 Seaborn Bug，手工在绝对坐标系下绘制 Boxplot ⭐
        # 提取绘图数据：仅保留 'Entry', 'Corner', 'Exit'
        # phases_list = ['Overall', 'Entry', 'Corner', 'Exit']
        phases_list = ['Entry', 'Corner', 'Exit']
        HUE_OFFSETS = [-0.3, -0.1, 0.1, 0.3] # 4个类别的精准几何坐标偏移量
        for p_idx, phase in enumerate(phases_list):
            for c_idx in range(4):
                group_data = df_phase[(df_phase['Phase'] == phase) & (df_phase['Cluster'] == c_idx)][metric].dropna()
                if not group_data.empty:
                    pos = p_idx + HUE_OFFSETS[c_idx]
                    ax.boxplot(group_data, positions=[pos], widths=0.12, 
                               patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.2, zorder=10),
                               medianprops=dict(color='black', linewidth=1.5, zorder=10),
                               whiskerprops=dict(color='black', linewidth=1.2, zorder=10),
                               capprops=dict(color='black', linewidth=1.2, zorder=10))
        # 3. 分组散点图 (利用 dodge 完美对齐)
        sns.stripplot(x='Phase', y=metric, hue='Cluster', data=df_phase, 
                      palette=PALETTE, dodge=True, size=3, jitter=0.1, alpha=0.7, legend=False, ax=ax)
        
        # 添加阶段显著性标注
        draw_grouped_significance_brackets(ax, df_phase, metric, df_dunn_phase, phases_list)

        # 添加漂亮的图例
        handles = [mpatches.Patch(color=PALETTE[i],alpha=0.5, label=f'簇 {i+1}') for i in range(4)]
        ax.legend(handles=handles, title='聚类簇', loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=True)

        # ⭐ 核心修复：强制重置并锁死 X 轴刻度和标签，去掉 xlabel
        ax.set_xlabel('', fontsize=14)
        ax.set_ylabel(phase_ylabels[metric], fontsize=14)
        # ax.set_xticks([0, 1, 2, 3])
        # ax.set_xticklabels(['Overall', 'Entry', 'Corner', 'Exit'])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['$\\mathrm{Entry}$', '$\\mathrm{Corner}$', '$\\mathrm{Exit}$'])

        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()
        fig_name = f'Fig{9 + idx}_{metric}_Phase_Distribution.png'
        fig.savefig(os.path.join(save_dir, fig_name), dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"  - 成功保存 {fig_name}")

    print(f"🎉 恭喜！所有高级动力学分析与 IEEE 制图任务全部完成！")

if __name__ == '__main__':
    main()