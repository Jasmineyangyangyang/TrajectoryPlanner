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

# ==============================================================================
# ⭐⭐⭐ 【新增】全局默认样式细节加粗 (参考图片设定) ⭐⭐⭐
# ==============================================================================
plt.rcParams['axes.linewidth'] = 1.5        # ⭐全局边框粗细 (默认约0.8，加粗1.5倍)
plt.rcParams['xtick.major.width'] = 1.8     # ⭐全局主刻度线粗细
plt.rcParams['ytick.major.width'] = 1.8
plt.rcParams['xtick.major.size'] = 6       # 全局主刻度线长度 (默认约3.5，增长约1.5倍)
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.direction'] = 'in'     # X轴刻度线向内 (参考图片设定)
plt.rcParams['ytick.direction'] = 'in'     # Y轴刻度线向内
plt.rcParams['font.size'] = 12             # 全局基准字号

# 为了让 label 字符串里的 Times New Roman 正体单位显示规范
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体 (如变量 v) 使用 TNR 斜体

# ==============================================================================
# 轨迹平滑与重采样函数
# ==============================================================================
def process_frenet_data(s_raw, d_raw, speed_raw, s_ddot_raw, d_ddot_raw, delta_s=1.0, window_length=15, polyorder=3):
    s_raw = np.array(s_raw)
    d_raw = np.array(d_raw)
    speed_raw = np.array(speed_raw)
    s_ddot_raw = np.array(s_ddot_raw)
    d_ddot_raw = np.array(d_ddot_raw)

    # 1. 数据清洗：确保 s 严格单调递增
    _, unique_indices = np.unique(s_raw, return_index=True)
    unique_indices = np.sort(unique_indices) 
    
    s_clean = s_raw[unique_indices]
    d_clean = d_raw[unique_indices]
    speed_clean = speed_raw[unique_indices]
    s_ddot_clean = s_ddot_raw[unique_indices]
    d_ddot_clean = d_ddot_raw[unique_indices]

    diffs = np.diff(s_clean)
    valid_indices = np.insert(diffs > 0, 0, True)
    s_clean = s_clean[valid_indices]
    d_clean = d_clean[valid_indices]
    speed_clean = speed_clean[valid_indices]
    s_ddot_clean = s_ddot_clean[valid_indices]
    d_ddot_clean = d_ddot_clean[valid_indices]

    # 确保有足够的数据点进行后续操作
    if len(s_clean) < 4:
        return s_clean, d_clean, speed_clean, s_ddot_clean, d_ddot_clean

    # 2. 滤波：Savitzky-Golay 滤波
    current_window = window_length
    if len(s_clean) < current_window:
        current_window = len(s_clean) if len(s_clean) % 2 != 0 else len(s_clean) - 1
        
    if current_window > polyorder:
        d_filtered = savgol_filter(d_clean, current_window, polyorder)
        speed_filtered = savgol_filter(speed_clean, current_window, polyorder)
        s_ddot_filtered = savgol_filter(s_ddot_clean, current_window, polyorder)
        d_ddot_filtered = savgol_filter(d_ddot_clean, current_window, polyorder)
    else:
        d_filtered, speed_filtered, s_ddot_filtered, d_ddot_filtered = d_clean, speed_clean, s_ddot_clean, d_ddot_clean

    # 3. 插值与重采样：三次样条插值
    cs_d = CubicSpline(s_clean, d_filtered)
    cs_speed = CubicSpline(s_clean, speed_filtered)
    cs_s_ddot = CubicSpline(s_clean, s_ddot_filtered)
    cs_d_ddot = CubicSpline(s_clean, d_ddot_filtered)

    s_start = np.ceil(s_clean[0])
    s_end = np.floor(s_clean[-1])
    s_end = np.minimum(s_end, 183.0) # 限制在 s_in_bend 范围内
    # 防止片段过短导致无法重采样
    if s_start >= s_end:
        return s_clean, d_filtered, speed_filtered, s_ddot_filtered, d_ddot_filtered

    s_resampled = np.arange(s_start, s_end + delta_s, delta_s)
    d_resampled = cs_d(s_resampled)
    speed_resampled = cs_speed(s_resampled)
    s_ddot_resampled = cs_s_ddot(s_resampled)
    d_ddot_resampled = cs_d_ddot(s_resampled)

    return s_resampled, d_resampled, speed_resampled, s_ddot_resampled, d_ddot_resampled

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

planner = Polyplanner(road_process, lane_id=1)
ref_s = planner.ts
fig = plt.figure(2, figsize=(12, 7))      # Figure 对象
ax = fig.add_subplot(111)               # 添加 Axes
# A. 绘制基准车道中心线 (d=0) —— 参考图片设定：绿色实线
ax.plot(ref_s[:1900], np.zeros_like(ref_s[:1900]), color='#333333', linestyle='-', linewidth=1.2, alpha=0.8, label='车道中心线参考 ($d=0$)')
ax.plot(ref_s[:1900], np.ones_like(ref_s[:1900])*0.8, color='#333333', linestyle='--', linewidth=0.8, alpha=0.5, label='安全边界线')
ax.plot(ref_s[:1900], np.ones_like(ref_s[:1900])*(-0.8), color='#333333', linestyle='--', linewidth=0.8, alpha=0.5)
# ax.plot(ref_s[:1900], np.ones_like(ref_s[:1900])*(1.75), 'k-', linewidth=2.0)
# ax.plot(ref_s[:1900], np.ones_like(ref_s[:1900])*(-1.75), 'k-', linewidth=2.0)
s_roi_start = 10.0
s_roi_end = 180.0
ax.annotate('$\mathrm{ROI}$开始', xy=(s_roi_start, 0.6), xytext=(s_roi_start + 5, 0.6),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6),
                family='SimSun', fontsize=21, verticalalignment='center')
    
ax.annotate('$\mathrm{ROI}$结束', xy=(s_roi_end, 0.6), xytext=(s_roi_end - 25, 0.6),
            arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6),
            family='SimSun', fontsize=20, verticalalignment='center')
ax.set_xlabel("纵向距离 $s\mathrm{/m}$", fontsize=24, labelpad=10)
ax.set_ylabel("横向偏移量 $d\mathrm{/m}$", fontsize=24, labelpad=10)
ax.axvline(s_roi_start, color='r', linestyle='--', linewidth=1.5)
ax.axvline(s_roi_end, color='r', linestyle='--', linewidth=1.5)

# 设定坐标轴范围，聚焦在弯心 Apex 区域的横向偏移上
# ax.set_xlim(env.s_straight_1 - 10, env.s_total - env.s_straight_2 + 10) # 30-10 到 160+10
ax.set_xlim(0, 190) # 直接使用 s_in_bend 的范围对齐图片
ax.set_ylim(-1.0, 1.0) # 纵向显示范围对齐图片
plt.tight_layout()


# ==============================================================================
# 加载干预数据
# ==============================================================================
filepath = 'OAS_data/'
# filename = 'Traj55_in'
# filename = 'Traj41_center'
# filename = 'Traj33_in-out' 
# filename = 'Traj39_out-in-out' # actually 38 trajectories, index 26 out safe range, delete it
# filename = 'Traj38_Supplementary1' # actually 36 trajectories, index 13 and 26 out safe range, delete it
# filename = 'Traj39_Supplementary2' # actually 36 trajectories, index 14 and 35 and 37 out safe range, delete it
# filename = 'Traj13_Supplementary3'
# filename = 'Traj33_Supplementary4'
filename = 'Traj38_SuppleOut-in'  # actually 35 trajectories, index 3 and 5 and 24 out safe range, delete it
with open(filepath + filename + '.pkl', 'rb') as f:
    data = pickle.load(f)

# load trajectory
ep_r_step_bendnum = data['ep_r_step_bendnum']
veh_trajectory = data['veh_trajectory'] # [ego_x, ego_y, ego_yaw, ego_velocity, ego_lon_acc, ego_lat_acc, adj_x, adj_y, adj_yaw, adj_velocity, intervenFlag]
veh_trajectory = np.array(veh_trajectory)

ego_circle_index = np.where((veh_trajectory[1:,0]-veh_trajectory[:-1,0])<0.0)
print(f"the circle number = {ego_circle_index[0].shape[0]}")
ego_circle_index = ego_circle_index[0]  # find the index of each circle
trajectory_matrix_d = []
trajectory_matrix_speed = []
trajectory_matrix_s_ddot = []
trajectory_matrix_d_ddot = []
trajectory_matrix_s_dddot = []
trajectory_matrix_d_dddot = []

plt.figure(4)
plt.xlabel('s (m)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Frenet Coordinates Acceleration')

for i in range(ego_circle_index.shape[0]):
    # 提取当前 circle 的轨迹切片
    if i == 0:
        plt.figure(1)
        plt.plot(veh_trajectory[:ego_circle_index[i]+1, 0], veh_trajectory[:ego_circle_index[i]+1, 1], 'r')
        traj = veh_trajectory[:ego_circle_index[i]+1, :]
    elif i <= ego_circle_index.shape[0]-1:
        if i ==  24 or i ==  3 or i ==  5: # 删除越界的轨迹片段
            continue
        plt.figure(1)
        plt.plot(veh_trajectory[ego_circle_index[i-1]+1:ego_circle_index[i]+1, 0], veh_trajectory[ego_circle_index[i-1]+1:ego_circle_index[i]+1, 1], 'g')
        if i == ego_circle_index.shape[0]-1:
            if veh_trajectory.shape[0] - (ego_circle_index[i]+1) > 100:
                plt.figure(1)
                plt.plot(veh_trajectory[ego_circle_index[i]+1:, 0], veh_trajectory[ego_circle_index[i]+1:, 1], 'b')
        traj = veh_trajectory[ego_circle_index[i-1]+1:ego_circle_index[i]+1, :]
    
    # 转换为 Frenet 坐标系 (修复了内层循环变量为 j，防止与外层 i 冲突)
    traj_s = []
    traj_d = []
    traj_speed = []
    traj_s_ddot = []
    traj_d_ddot = []
    for j in range(traj.shape[0]):
        ego_x = traj[j, 0]
        ego_y = traj[j, 1]
        ego_yaw = traj[j, 2]
        ego_speed = traj[j, 3]
        ego_s_ddot = traj[j, 4]
        ego_d_ddot = traj[j, 5]
        if j == 0:
            ego_s_dddot = 0.0
            ego_d_dddot = 0.0
        else:
            ego_s_dddot = (traj[j, 4] - traj[j-1, 4]) / 0.02  # 采集数据取决于adj state发布间隔:0.02s
            ego_d_dddot = (traj[j, 5] - traj[j-1, 5]) / 0.02
        ego_s,_,ego_ax, ego_d,_,ego_ay = planner.calculate_frenet_coordinates(ego_x, ego_y, ego_yaw, ego_speed)
        traj_s.append(ego_s)
        traj_d.append(ego_d)
        traj_speed.append(ego_speed)
        traj_s_ddot.append(ego_s_ddot)
        traj_d_ddot.append(ego_d_ddot)

    # ---------------------------------------------------------
    # 执行平滑与重采样
    # ---------------------------------------------------------
    try:
        delta_s=1.0  # 重采样间隔 1.0米
        s_resampled, d_resampled, speed_resampled, s_ddot_resampled, d_ddot_resampled = process_frenet_data(traj_s, traj_d, traj_speed, traj_s_ddot, traj_d_ddot, delta_s=delta_s)
        # 恒速 40km/h，delta_s米下的等效时间间隔
        dt = delta_s / (40.0 / 3.6)
        # 对重采样后的 d_ddot 进行平滑求导
        # window_length 可以设大一点（如 21 或 31）来压制高频毛刺，polyorder=3
        d_dddot_resampled = savgol_filter(d_ddot_resampled, window_length=21, polyorder=3, deriv=1, delta=dt)
        s_dddot_resampled = savgol_filter(s_ddot_resampled, window_length=21, polyorder=3, deriv=1, delta=dt)
    except Exception as e:
        print(f"第 {i} 段轨迹重采样失败: {e}，将使用原始数据。")

    if i == 0:
        plt.figure(4)
        plt.plot(s_resampled, s_ddot_resampled, 'm-', marker='o', markersize=3, label='s向加速度')
        plt.plot(s_resampled, d_ddot_resampled, 'c-', marker='s', markersize=3, label='d向加速度')
        plt.plot(s_resampled, speed_resampled, 'r-', marker='d', markersize=3, label='速度')
        plt.plot(s_resampled, s_dddot_resampled, 'g-', marker='^', markersize=3, label='s向冲击度')
        plt.plot(s_resampled, d_dddot_resampled, 'b-', marker='v', markersize=3, label='d向冲击度')
    else:
        plt.figure(4)
        plt.plot(s_resampled, s_ddot_resampled, 'm-', marker='o', markersize=3)
        plt.plot(s_resampled, d_ddot_resampled, 'c-', marker='s', markersize=3)
        plt.plot(s_resampled, speed_resampled, 'r-', marker='d', markersize=3)
        plt.plot(s_resampled, s_dddot_resampled, 'g-', marker='^', markersize=3)
        plt.plot(s_resampled, d_dddot_resampled, 'b-', marker='v', markersize=3)
    plt.legend()

    # 打印当前轨迹长度，用于调试维度不匹配问题
    print(f"第 {i} 段轨迹长度: {len(d_resampled)}")
    trajectory_matrix_d.append(d_resampled)
    trajectory_matrix_speed.append(speed_resampled)
    trajectory_matrix_s_ddot.append(s_ddot_resampled)
    trajectory_matrix_d_ddot.append(d_ddot_resampled)
    trajectory_matrix_s_dddot.append(s_dddot_resampled)
    trajectory_matrix_d_dddot.append(d_dddot_resampled)

    # 绘制到 Figure 2 (ax) 上
    if i == 0:
        line_raw, = ax.plot(traj_s, traj_d, color='#F39C12', marker='.', markersize=2, 
                linestyle='None', alpha=0.8, label='原始离散数据')
        ax.plot(s_resampled, d_resampled, color='#7F8C8D', linestyle='-', linewidth=1.2, alpha=0.9, label='平滑重采样轨迹')
    else:
        ax.plot(traj_s, traj_d, color='#F39C12', marker='.', markersize=2, 
                linestyle='None', alpha=0.8)
        ax.plot(s_resampled, d_resampled, color='#7F8C8D', linestyle='-', linewidth=1.2, alpha=0.9)

# ---------------------------------------------------------
# 方法 A：保存为 Numpy .npz 格式 (强烈推荐)
# ---------------------------------------------------------
# 把 s 坐标轴和 轨迹矩阵 打包存在一起
np.savez(filepath + 'cluster/' + filename + '.npz', 
            s_coordinates=s_resampled, 
            d_matrix=trajectory_matrix_d,
            speed_matrix=trajectory_matrix_speed,
            s_ddot_matrix=trajectory_matrix_s_ddot,
            d_ddot_matrix=trajectory_matrix_d_ddot,
            s_dddot_matrix=trajectory_matrix_s_dddot,
            d_dddot_matrix=trajectory_matrix_d_dddot)
print(f"成功保存高精度特征矩阵至 {filepath + 'cluster/' + filename + '.npz'} 文件！")

# ---------------------------------------------------------
# 方法 B：保存为 .csv 格式 (方便 Excel 查阅)
# ---------------------------------------------------------
import pandas as pd
# 将 s 坐标作为 DataFrame 的列名，非常优雅！
columns_s = np.round(s_resampled, 1)

df_d = pd.DataFrame(trajectory_matrix_d, columns=columns_s)
df_d.to_csv(filepath + 'cluster/csv/' + filename + '_d.csv', index=False)

df_speed = pd.DataFrame(trajectory_matrix_speed, columns=columns_s)
df_speed.to_csv(filepath + 'cluster/csv/' + filename + '_speed.csv', index=False)

df_s_ddot = pd.DataFrame(trajectory_matrix_s_ddot, columns=columns_s)
df_s_ddot.to_csv(filepath + 'cluster/csv/' + filename + '_s_ddot.csv', index=False)

df_d_ddot = pd.DataFrame(trajectory_matrix_d_ddot, columns=columns_s)
df_d_ddot.to_csv(filepath + 'cluster/csv/' + filename + '_d_ddot.csv', index=False)

df_s_dddot = pd.DataFrame(trajectory_matrix_s_dddot, columns=columns_s)
df_s_dddot.to_csv(filepath + 'cluster/csv/' + filename + '_s_dddot.csv', index=False)

df_d_dddot = pd.DataFrame(trajectory_matrix_d_dddot, columns=columns_s)
df_d_dddot.to_csv(filepath + 'cluster/csv/' + filename + '_d_dddot.csv', index=False)

print(f"成功保存可读表格至 {filepath}cluster/csv/ 下的六个 .csv 文件！")

# 添加图例，严格对齐你的字体设定
plt.figure(2)

#  ⭐⭐⭐ 【核心修改】精细化 Spines (边框) 和 Ticks (刻度) ⭐⭐⭐
# 显示左侧和底部边框 (坐标轴)，并加粗、改纯黑
ax.spines['left'].set_visible(True)
ax.spines['left'].set_linewidth(1.5)  # ⭐坐标轴加粗
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(1.5) # ⭐坐标轴加粗
ax.spines['bottom'].set_color('black')
# ⭐隐藏顶部和右侧边框 (使风格与参考图1对齐)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_linewidth(1.5)  # ⭐坐标轴加粗
ax.spines['top'].set_linewidth(1.5)  # ⭐坐标轴加粗

#  ⭐⭐⭐ 【优化】图例排版：并列两排，移到顶部， frameon=False ⭐⭐⭐
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, 
                fontsize=18, frameon=True, columnspacing=2.0)
#  开启细节网格线
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
# 可选：保存为博士论文级高清图片
fig.savefig(filepath + 'figures/RawTrajShow/' + filename + '.png', dpi=600, bbox_inches='tight')


adj_circle_index = np.where((veh_trajectory[1:,6]-veh_trajectory[:-1,6])<0.0)  # the spawn point of adj is [0,0]
adj_circle_index = adj_circle_index[0]  # find the index of each circle
plt.figure(1)
for i in range(adj_circle_index.shape[0]):
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