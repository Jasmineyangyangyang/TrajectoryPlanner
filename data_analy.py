import pickle
import matplotlib.pyplot as plt
import numpy as np
from global_road import natural_road_load
from polyplan_States_cost import Polyplanner

# --- 新增的 scipy 依赖 ---
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

# ==============================================================================
# 全局图表样式配置 (博士论文制图强迫症级配置)
# ==============================================================================
# 优先使用 Times New Roman (英文字母与数字)，遇到中文时自动后退使用 SimSun (宋体)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC'] 
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
plt.rcParams['xtick.direction'] = 'in'     # X轴刻度线向内 (参考图片设定)
plt.rcParams['ytick.direction'] = 'in'     # Y轴刻度线向内
plt.rcParams['font.size'] = 12             # 全局基准字号

# 为了让 label 字符串里的 Times New Roman 正体单位显示规范
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

# ==============================================================================
# 轨迹平滑与重采样函数
# ==============================================================================
def process_frenet_data(s_raw, d_raw, delta_s=1.0, window_length=15, polyorder=3):
    s_raw = np.array(s_raw)
    d_raw = np.array(d_raw)

    # 1. 数据清洗：确保 s 严格单调递增
    _, unique_indices = np.unique(s_raw, return_index=True)
    unique_indices = np.sort(unique_indices) 
    
    s_clean = s_raw[unique_indices]
    d_clean = d_raw[unique_indices]

    diffs = np.diff(s_clean)
    valid_indices = np.insert(diffs > 0, 0, True)
    s_clean = s_clean[valid_indices]
    d_clean = d_clean[valid_indices]

    # 确保有足够的数据点进行后续操作
    if len(s_clean) < 4:
        return s_clean, d_clean, d_clean

    # 2. 滤波：Savitzky-Golay 滤波
    current_window = window_length
    if len(s_clean) < current_window:
        current_window = len(s_clean) if len(s_clean) % 2 != 0 else len(s_clean) - 1
        
    if current_window > polyorder:
        d_filtered = savgol_filter(d_clean, current_window, polyorder)
    else:
        d_filtered = d_clean 

    # 3. 插值与重采样：三次样条插值
    cs = CubicSpline(s_clean, d_filtered)

    s_start = np.ceil(s_clean[0])
    s_end = np.floor(s_clean[-1])
    # 防止片段过短导致无法重采样
    if s_start >= s_end:
        return s_clean, d_filtered, d_filtered
        
    s_resampled = np.arange(s_start, s_end + delta_s, delta_s)
    d_resampled = cs(s_resampled)

    return s_resampled, d_resampled, d_filtered


# load bend
road_process = natural_road_load()
road = road_process.read_from_csv()
road_left = road[:,0:2]     # [x, y]
road_right = road[:,2:4]    # [x, y]
road_center = road[:, 4:6]  # [x, y]

plt.figure(1)
plt.gcf().canvas.mpl_connect(
    'key_release_event',
    lambda event: [exit(0) if event.key == 'escape' else None])
plt.plot(road_left[:, 0], road_left[:, 1], 'k')
plt.plot(road_right[:, 0], road_right[:, 1], 'k')
plt.plot(road_center[:, 0], road_center[:, 1], 'k')

plt.plot([402.02,402.21], [-237.79,-245.29], 'b')
plt.plot([574.93,570.15] ,[-300.92,-306.70], 'b')
# for stopping simulation with the esc key.

planner = Polyplanner(road_process, lane_id=1)
ref_s = planner.ts
fig = plt.figure(2, figsize=(12, 7))      # Figure 对象
ax = fig.add_subplot(111)               # 添加 Axes
# A. 绘制基准车道中心线 (d=0) —— 参考图片设定：绿色实线
ax.plot(ref_s[:1900], np.zeros_like(ref_s[:1900]), 'g-', linewidth=2.0, label='车道中心线')
ax.plot(ref_s[:1900], np.ones_like(ref_s[:1900])*0.8, 'g--', linewidth=2.0, label='安全边界')
ax.plot(ref_s[:1900], np.ones_like(ref_s[:1900])*(-0.8), 'g--', linewidth=2.0)
s_roi_start = 10.0
s_roi_end = 180.0
ax.annotate('$\mathrm{ROI}$开始', xy=(s_roi_start, 0.6), xytext=(s_roi_start + 5, 0.6),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6),
                family='SimSun', fontsize=21, verticalalignment='center')
    
ax.annotate('$\mathrm{ROI}$结束', xy=(s_roi_end, 0.6), xytext=(s_roi_end - 25, 0.6),
            arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6),
            family='SimSun', fontsize=20, verticalalignment='center')
ax.set_xlabel("纵向距离 $\mathrm{d/m}$", fontsize=24, labelpad=10)
ax.set_ylabel("横向偏移量 $\mathrm{s/m}$", fontsize=24, labelpad=10)
ax.axvline(s_roi_start, color='r', linestyle='--', linewidth=1.5)
ax.axvline(s_roi_end, color='r', linestyle='--', linewidth=1.5)

# 设定坐标轴范围，聚焦在弯心 Apex 区域的横向偏移上
# ax.set_xlim(env.s_straight_1 - 10, env.s_total - env.s_straight_2 + 10) # 30-10 到 160+10
ax.set_xlim(0, 190) # 直接使用 s_in_bend 的范围对齐图片
ax.set_ylim(-1.0, 1.0) # 纵向显示范围对齐图片，容纳 -0.4借道 和 1.2切弯
plt.tight_layout()
    

with open('./OAS_data/hug_data_20260402_22-07-29_road375_yjx_inter30traj.pkl', 'rb') as f:
    data = pickle.load(f)

# load trajectory
ep_r_step_bendnum = data['ep_r_step_bendnum']
veh_trajectory = data['veh_trajectory'] # [ego_x, ego_y, ego_yaw, ego_velocity, ego_lon_acc, ego_lat_acc, adj_x, adj_y, adj_yaw, adj_velocity, intervenFlag]
veh_trajectory = np.array(veh_trajectory)

ego_circle_index = np.where((veh_trajectory[1:,0]-veh_trajectory[:-1,0])<0.0)
print(f"the circle number = {ego_circle_index[0].shape[0]}")
ego_circle_index = ego_circle_index[0]  # find the index of each circle
trajectory_matrix = []
plt.figure(1)
for i in range(30):
    # 提取当前 circle 的轨迹切片
    if i == 0:
        plt.plot(veh_trajectory[:ego_circle_index[i]+1, 0], veh_trajectory[:ego_circle_index[i]+1, 1], 'r')
        traj = veh_trajectory[:ego_circle_index[i]+1, :]
    elif i <= ego_circle_index.shape[0]-1:
        plt.plot(veh_trajectory[ego_circle_index[i-1]+1:ego_circle_index[i]+1, 0], veh_trajectory[ego_circle_index[i-1]+1:ego_circle_index[i]+1, 1], 'g')
        if i == ego_circle_index.shape[0]-1:
            if veh_trajectory.shape[0] - (ego_circle_index[i]+1) > 100:
                plt.plot(veh_trajectory[ego_circle_index[i]+1:, 0], veh_trajectory[ego_circle_index[i]+1:, 1], 'b')
        traj = veh_trajectory[ego_circle_index[i-1]+1:ego_circle_index[i]+1, :]
    
    # 转换为 Frenet 坐标系 (修复了内层循环变量为 j，防止与外层 i 冲突)
    traj_s = []
    traj_d = []
    for j in range(traj.shape[0]):
        ego_x = traj[j, 0]
        ego_y = traj[j, 1]
        ego_yaw = traj[j, 2]
        ego_speed = traj[j, 3]
        ego_s,_,_, ego_d,_,_ = planner.calculate_frenet_coordinates(ego_x, ego_y, ego_yaw, ego_speed)
        traj_s.append(ego_s)
        traj_d.append(ego_d)
    
    # ---------------------------------------------------------
    # 执行平滑与重采样
    # ---------------------------------------------------------
    try:
        s_resampled, d_resampled, _ = process_frenet_data(traj_s, traj_d, delta_s=1.0)
        plot_s, plot_d = s_resampled, d_resampled
    except Exception as e:
        print(f"第 {i} 段轨迹重采样失败: {e}，将使用原始数据。")
        plot_s, plot_d = traj_s, traj_d
    trajectory_matrix.append(plot_d)

    # 绘制到 Figure 2 (ax) 上
    if i == 0:
        ax.plot(traj_s, traj_d, color='#F39C12', marker='.', markersize=2, 
                linestyle='None', alpha=0.3, label='原始离散数据')
        ax.plot(plot_s, plot_d, color='#7F8C8D', linestyle='-', linewidth=1.2, alpha=0.8, label='平滑重采样轨迹')
    else:
        ax.plot(traj_s, traj_d, color='#F39C12', marker='.', markersize=2, 
                linestyle='None', alpha=0.3)
        ax.plot(plot_s, plot_d, color='#7F8C8D', linestyle='-', linewidth=1.2, alpha=0.8)

# ---------------------------------------------------------
# 方法 A：保存为 Numpy .npz 格式 (强烈推荐)
# ---------------------------------------------------------
# 把 s 坐标轴和 轨迹矩阵 打包存在一起
np.savez('./OAS_data/trajectory_clustering_dataset.npz', 
            s_coordinates=plot_s, 
            d_matrix=trajectory_matrix)
print("成功保存高精度特征矩阵至 .npz 文件！")

# ---------------------------------------------------------
# 方法 B：保存为 .csv 格式 (方便 Excel 查阅)
# ---------------------------------------------------------
import pandas as pd
# 将 s 坐标作为 DataFrame 的列名，非常优雅！
df_trajectories = pd.DataFrame(trajectory_matrix, columns=np.round(plot_s, 1))
df_trajectories.to_csv('./OAS_data/trajectory_clustering_dataset.csv', index=False)
print("成功保存可读表格至 .csv 文件！")

# 添加图例，严格对齐你的字体设定
plt.figure(2)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=20)
# 注意：要把 loc 改为 'lower center'，这样图例的底部才会对齐在 bbox 的 1.02 位置，从而完全被推到图外
# ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, 
#           fontsize=16,          # 字号稍微调小一点，配合单行排版
#           frameon=False,        # 去除边框，显得极其干净
#           columnspacing=1.5,    # 调整列间距，避免太散
#           handletextpad=0.5)    # 缩小线条和文字之间的距离
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, 
          fontsize=18, 
        #   frameon=False,        # 灵魂操作：去除边框
          columnspacing=3.0)    # 稍微拉开左右两列的间距，对齐更美观
# 开启细节网格线 (参考图片设定：半透明虚线)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# 可选：保存为博士论文级高清图片
fig.savefig('./figures/trajectory_sampling_distribution_30.png', dpi=600, bbox_inches='tight')


adj_circle_index = np.where((veh_trajectory[1:,6]-veh_trajectory[:-1,6])<0.0)  # the spawn point of adj is [0,0]
adj_circle_index = adj_circle_index[0]  # find the index of each circle
plt.figure(1)
for i in range(30):
    if i == 0:
        plt.plot(veh_trajectory[:adj_circle_index[i]+1, 6], veh_trajectory[:adj_circle_index[i]+1, 7], 'r')  # [adj_x, adj_y]
    elif i < adj_circle_index.shape[0]-1:
        plt.plot(veh_trajectory[adj_circle_index[i]+2:adj_circle_index[i+1]+1, 6], veh_trajectory[adj_circle_index[i]+2:adj_circle_index[i+1]+1, 7], 'g')  # [ego_x, ego_y]
    else:
        if veh_trajectory.shape[0] - (adj_circle_index[i]+1) > 100:
            plt.plot(veh_trajectory[adj_circle_index[i]+1:veh_trajectory.shape[0], 6], veh_trajectory[adj_circle_index[i]+1:veh_trajectory.shape[0], 7], 'b')  

plt.title("trajectory")


plt.figure(3)
plt.gcf().canvas.mpl_connect(
    'key_release_event',
    lambda event: [exit(0) if event.key == 'escape' else None])
plt.plot(veh_trajectory[:, 3], 'r', label='ego')
plt.plot(veh_trajectory[:, 9], 'g', label = 'adjacent')
plt.plot(adj_circle_index, veh_trajectory[0, 9]*np.ones_like(adj_circle_index), 'k.', label = 'bend flag')
plt.ylim([10, 15])
plt.title("velocity")
plt.legend()
plt.show()

'''
后续被注释掉的在线训练代码部分保持不变，为节省空间在此不重复打印。
'''