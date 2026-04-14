import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. 配置符合要求的 rcParams
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC', 'STSong'] # 宋体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 14
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

# 2. 数据处理与重命名
file_path = './Eye_data/DriverRank.csv'
df = pd.read_csv(file_path, index_col=0)

mapping = {
    'traject1': '内偏', 'traject2': '簇4', 'traject3': '簇3',
    'traject4': '簇1', 'traject5': '居中', 'traject6': '外偏',
    'trajcet7': '簇2', 'traject7': '簇2'
}
df.index = [mapping.get(idx, idx) for idx in df.index]

# 3. 绘制热力图
plt.figure(figsize=(12, 6))
ax = sns.heatmap(df, 
                 annot=True, 
                 cmap="RdYlGn_r", 
                 linewidths=0.8, 
                 linecolor='white',
                 cbar_kws={'label': '偏好排序 (1为最优)'})

# plt.title('驾驶员对不同运动描述簇轨迹的接受度排序', fontsize=16, pad=20)
plt.xlabel('驾驶员编号 $\mathrm{(Driver ID)}$')
plt.ylabel('代表路径', fontname='SimSun')

# 强制将数字刻度设置为 Times New Roman
for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')

plt.savefig('./Eye_data/Driver_Acceptance_Heatmap_Final.png', dpi=300, bbox_inches='tight')


# 2. 绘堆叠条形图
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("RdYlGn_r", 7)
rank_dist = df.apply(pd.Series.value_counts, axis=1).fillna(0)
rank_dist = rank_dist[[1, 2, 3, 4, 5, 6, 7]] # 确保排序正确


rank_dist.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='black', linewidth=0.8)

# 3. 标签与图例
# plt.title('各代表路径的偏好位次频率分布', fontsize=16, pad=20)
plt.xlabel('代表路径', fontname='SimSun')
plt.ylabel('驾驶员人数 $\mathrm{(Counts)}$', fontname='SimSun')
plt.legend(['第1名', '第2名', '第3名', '第4名', '第5名', '第6名', '第7名'], 
           title='偏好位次', bbox_to_anchor=(1.02, 1))

plt.savefig('./Eye_data/Trajectory_Rank_StackedBar_Final.png', dpi=300, bbox_inches='tight')