import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ==============================================================================
# 全局图表样式配置 (满足学术期刊标准)
# ==============================================================================
plt.rcParams['font.family'] = ['sans-serif', 'Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体 (如变量 v) 使用 TNR 斜体


def main():
    # ==========================================================================
    # 1. 路径与数据加载
    # ==========================================================================
    # 兼容你提到的 Table1/2 或者我之前代码默认生成的 Table3/4 命名
    base_dir = 'OAS_data/figures/comfort_analysis/'
    
    kw_path = os.path.join(base_dir, 'Table1_Phase_Kruskal_Wallis.csv')
    dunn_path = os.path.join(base_dir, 'Table2_Phase_Dunn_Posthoc.csv')
    
    # 智能 fallback：如果找不到 Table1/2，尝试寻找 Table3/4
    if not os.path.exists(kw_path):
        kw_path = os.path.join(base_dir, 'Table3_Phase_Kruskal_Wallis.csv')
        dunn_path = os.path.join(base_dir, 'Table4_Phase_Dunn_Posthoc.csv')

    print(f"📂 正在加载 KW 检验数据: {kw_path}")
    print(f"📂 正在加载 Dunn 检验数据: {dunn_path}")
    
    df_kw = pd.read_csv(kw_path)
    df_dunn = pd.read_csv(dunn_path)

    # ==========================================================================
    # 2. 基础配置
    # ==========================================================================
    color_stops = [
    (0.00, '#6FAED9'),   # 柔和蓝（低p，显著）
    (0.25, '#A9CCE3'),
    (0.50, '#FBFCFC'),   # 几乎纯白（干净过渡）
    (0.75, '#F5C6C6'),
    (1.00, '#D98880')    # 柔和红（高p，不显著）
    ]
    ieee_cmap = LinearSegmentedColormap.from_list('ieee_pvalue', color_stops)
    phases = ['Overall', 'Entry', 'Corner', 'Exit']
    metrics = ['max_abs_d_ddot', 'max_abs_d_dddot', 'max_abs_s_ddot', 'max_abs_s_dddot']
    cluster_labels = ['簇 1', '簇 2', '簇 3', '簇 4']
    
    # 创建专门保存热力图的文件夹
    heatmap_dir = os.path.join(base_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    print(f"📁 热力图将保存在: {heatmap_dir}\n")

    # ==========================================================================
    # 3. 循环生成 16 张 Heatmap
    # ==========================================================================
    count = 0
    for metric in metrics:
        for phase in phases:
            # --- 提取对应的 KW 检验 p-value ---
            kw_row = df_kw[(df_kw['Phase'] == phase) & (df_kw['Indicator'] == metric)]
            if kw_row.empty:
                continue
            kw_p = kw_row.iloc[0]['p-value']
            
            # --- 开始绘图 ---
            fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
            # 去掉刻度线
            ax.tick_params(axis='both', which='both', length=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#D0D0D0')   # 浅灰
                spine.set_linewidth(1.0)     # 细线

            # ==================================================================
            # 分支 A：KW 不显著 (p >= 0.05) -> 绘制标准 SCI 空白占位图
            # ==================================================================
            if kw_p >= 0.05:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
                # 隐藏刻度和坐标轴标签
                ax.set_xticks([])
                ax.set_yticks([])
                
                # 统一边框（更浅、更细一点更像期刊风格）
                for spine in ax.spines.values():
                    spine.set_color('#D0D0D0')
                    spine.set_linewidth(1.2)

                # 标题（阶段名）——稍微上移一点，留出层次
                ax.text(0.5, 0.68, phase.capitalize(),
                        ha='center', va='center',
                        fontsize=14, fontweight='semibold', color='black', family='Times New Roman')

                # KW结果（核心信息）
                ax.text(0.5, 0.48, f"KW p = {kw_p:.3f} (n.s.)",
                        ha='center', va='center',
                        fontsize=12, color='black', family='Times New Roman')

                # 说明文字（弱化，不抢信息）
                ax.text(0.5, 0.30, "No post-hoc test",
                        ha='center', va='center',
                        fontsize=10.5, color='#7A7A7A', style='italic', family='Times New Roman')

                # 去掉坐标轴（关键）
                ax.set_xlabel('')
                ax.set_ylabel('')

                status_msg = "[n.s. placeholder]"

            # ==================================================================
            # 分支 B：KW 显著 (p < 0.05) -> 绘制下三角热力图
            # ==================================================================
            else:
                # --- 提取对应的 Dunn 检验数据 ---
                dunn_rows = df_dunn[(df_dunn['Phase'] == phase) & (df_dunn['Indicator'] == metric)]
                
                # 初始化 4x4 的数据矩阵和文本矩阵 (默认填充 NaN)
                p_matrix = np.full((4, 4), np.nan)
                annot_matrix = np.full((4, 4), "", dtype=object)

                # 填充矩阵 (下三角逻辑)
                for _, row in dunn_rows.iterrows():
                    comp = row['Cluster comparison']
                    p_val = row['Adjusted p-value']
                    
                    # 从 "C1 vs C2" 中提取索引 (0, 1, 2, 3)
                    parts = comp.split(' vs ')
                    c_i = int(parts[0].replace('C', '')) - 1
                    c_j = int(parts[1].replace('C', '')) - 1
                    
                    # 强制填入下三角 (行坐标 > 列坐标)
                    r, c = max(c_i, c_j), min(c_i, c_j)
                    p_matrix[r, c] = p_val
                    
                    # 构造带 * 号的标注文本
                    if p_val < 0.05:
                        annot_matrix[r, c] = f"{p_val:.3f}*"
                    else:
                        annot_matrix[r, c] = f"{p_val:.3f}"

                # --- 生成下三角 Mask ---
                # np.triu 会把上三角（包含对角线）标记为 True，在 sns.heatmap 中设为 True 的地方会被隐藏
                mask = np.triu(np.ones_like(p_matrix, dtype=bool))

                # 绘制 Heatmap
                # cmap=ieee_cmap: 使用自定义颜色映射
                sns.heatmap(p_matrix, 
                            mask=mask, 
                            annot=annot_matrix, 
                            fmt="",                  # fmt="" 允许直接填入包含 * 的字符串
                            cmap=ieee_cmap, 
                            vmin=0, vmax=0.05,       # 统一色彩阈值
                            square=True, 
                            linewidths=0.5, 
                            cbar_kws={'label': 'p-value'}, 
                            ax=ax)
                # 标准答案
                cbar = ax.collections[0].colorbar
                cbar.ax.yaxis.label.set_fontname('Times New Roman')  # ✅ 只改这个
                # 设置行列标签
                ax.set_xticklabels(cluster_labels, rotation=45, ha='right')
                ax.set_yticklabels(cluster_labels, rotation=0)

                # --- 构造优雅的标题 (带 KW p-value) ---
                if kw_p < 0.001:
                    kw_str = "KW $\\mathrm{{p}} < 0.001$"
                else:
                    kw_str = f"KW $\\mathrm{{p}} = {kw_p:.3f}$"  # \mathrm{p}
                
                # 保持标题如 "Overall (KW p = 0.012)"，且首字母自动大写
                ax.set_title(f"{phase.capitalize()} ({kw_str})", pad=15, fontsize=14, family='Times New Roman')

            # 紧凑布局并保存
            plt.tight_layout()
            
            # 文件名全小写：例如 max_abs_d_ddot_entry.png
            save_name = f"{metric}_{phase.lower()}.png"
            save_path = os.path.join(heatmap_dir, save_name)
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            
            count += 1
            print(f"  ✅ 成功生成: {save_name}")

    print(f"\n🎉 恭喜！完美生成 {count} 张热力图！所有图片已保存在 '{heatmap_dir}' 目录下。")

if __name__ == '__main__':
    main()