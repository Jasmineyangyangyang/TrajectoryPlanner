import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==============================================================================
# 全局图表样式配置 (满足顶刊与博士论文要求)
# ==============================================================================
# 优先使用 Times New Roman (英文字母与数字)，遇到中文时自动后退使用 SimSun (宋体)
plt.rcParams['font.family'] =  ['SimSun'] 
plt.rcParams['font.sans-serif'] = ['SimSun'] 
plt.rcParams['font.serif'] = ['SimSun'] 
# 2. 极其关键：自定义数学公式引擎的字体，强制将其设为 TNR
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'         # 正体 (如单位) 使用 TNR
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体 (如变量 v) 使用 TNR 斜体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
plt.rcParams['font.size'] = 12             # 全局基准字号
plt.rcParams['xtick.direction'] = 'in'     # X轴刻度线向内
plt.rcParams['ytick.direction'] = 'in'     # Y轴刻度线向内


# ==============================================================================
# 1. 环境加载与路网处理模块 (保持不变)
# ==============================================================================
class CurveFlag():
    def __init__(self) -> None:
        self.cur_in_left = np.array([[402.02, -237.79]])
        self.cur_in_right = np.array([[402.214, -245.289]]) 
        self.cur_out_left = np.array([[574.932, -300.919]])
        self.cur_out_right = np.array([[569.962, -306.537]])

class natural_road_load():
    def __init__(self):   
        self.curvfg = CurveFlag()
        
    def read_from_csv(self, filepath='./'):
        road_filename = os.path.join(filepath, 'global_road.csv')
        self.road = []
        with open(road_filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                self.road.append([float(row['left_x']), float(row['left_y']), 
                                  float(row['right_x']), float(row['right_y']), 
                                  float(row['center_x']), float(row['center_y'])])
        self.road_trajectory = np.array(self.road)
        return self.road_trajectory
        
    def curve_in_check(self, x, y):
        in_bend_vec = self.curvfg.cur_in_right - self.curvfg.cur_in_left
        out_bend_vec = self.curvfg.cur_out_right - self.curvfg.cur_out_left
        vehile_vec_in = np.array([x, y]) - self.curvfg.cur_in_left
        vehile_vec_out = np.array([x, y]) - self.curvfg.cur_out_left
        # 放宽了一点边界，确保弯道检测更鲁棒
        if (380.0 <= x <= 600.) and (-380 <= y <= -200):
            if np.cross(vehile_vec_in, in_bend_vec) < 0 and np.cross(vehile_vec_out, out_bend_vec) > 0:
                return True
        return False

# ==============================================================================
# 2. Frenet QP 多模态规划器与速度解算
# ==============================================================================
class FrenetQP_MultiModePlanner:
    def __init__(self):
        self.W_dl = 10.0            # 一阶导惩罚 (航向)
        self.W_ddl = 800.0          # 二阶导惩罚 (曲率)
        self.W_dddl = 5000.0        # 三阶导惩罚 (Jerk冲击度)

    def generate_derivative_matrices(self, N):
        L1 = np.zeros((N-1, N))
        for i in range(N-1): L1[i, i], L1[i, i+1] = -1.0, 1.0
        L2 = np.zeros((N-2, N))
        for i in range(N-2): L2[i, i], L2[i, i+1], L2[i, i+2] = 1.0, -2.0, 1.0
        L3 = np.zeros((N-3, N))
        for i in range(N-3): L3[i, i], L3[i, i+1], L3[i, i+2], L3[i, i+3] = -1.0, 3.0, -3.0, 1.0
        return L1, L2, L3

    def plan(self, road_s, is_in_curve_flags, mode='center'):
        N = len(road_s)
        W_ref = np.ones(N) * 500.0  
        D_ref = np.zeros(N)
        
        curve_indices = np.where(is_in_curve_flags)[0]
        if len(curve_indices) > 0:
            idx_start = curve_indices[0]
            idx_end = curve_indices[-1]
            idx_apex = (idx_start + idx_end) // 2
            
            W_ref[idx_start:idx_end+1] = 0.0 
            
            if mode == 'center':
                W_ref[idx_start:idx_end+1] = 100.0
                D_ref[idx_start:idx_end+1] = 0.0
                
            elif mode == 'offset':
                W_ref[idx_start:idx_end+1] = 100.0
                D_ref[idx_start:idx_end+1] = 0.3 
                
            elif mode == 'out_in_out':
                W_ref[idx_start + 10] = 500.0
                D_ref[idx_start + 10] = -0.5  
                W_ref[idx_apex] = 1000.0
                D_ref[idx_apex] = 0.9  
                W_ref[idx_end - 10] = 500.0
                D_ref[idx_end - 10] = -0.5  

        L1, L2, L3 = self.generate_derivative_matrices(N)
        H = 2.0 * np.diag(W_ref) + \
            2.0 * self.W_dl * (L1.T @ L1) + \
            2.0 * self.W_ddl * (L2.T @ L2) + \
            2.0 * self.W_dddl * (L3.T @ L3)
        f = -2.0 * (W_ref * D_ref)

        D_opt = np.linalg.solve(H, -f)
        return D_opt

def calc_speed_profile(x, y, road_s, target_speed_kmh=60.0, max_lat_acc=2.5):
    dx = np.gradient(x, road_s)
    dy = np.gradient(y, road_s)
    ddx = np.gradient(dx, road_s)
    ddy = np.gradient(dy, road_s)
    
    kappa = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-8)
    kappa = gaussian_filter1d(kappa, sigma=5) 
    
    target_v_ms = target_speed_kmh / 3.6
    safe_v = np.sqrt(max_lat_acc / (np.abs(kappa) + 1e-5))
    v_profile = np.clip(safe_v, 0, target_v_ms)
    v_profile = gaussian_filter1d(v_profile, sigma=20)
    return v_profile * 3.6 

# ==============================================================================
# 3. 独立出图 (三张精美的中文论文配图)
# ==============================================================================
if __name__ == '__main__':
    # 加载与解算数据
    env = natural_road_load()
    road = env.read_from_csv('./')
    
    road_center = np.array([[(r[4]+r[2])/2.0, (r[5]+r[3])/2.0] for r in road])
    ds = np.hypot(np.diff(road_center[:,0]), np.diff(road_center[:,1]))
    road_s = np.concatenate(([0], np.cumsum(ds)))
    curve_flags = np.array([env.curve_in_check(x, y) for x, y in road_center])
    
    nx, ny = np.zeros_like(road_s), np.zeros_like(road_s)
    for i in range(len(road_s)):
        if i == len(road_s) - 1:
            dx, dy = road_center[i,0]-road_center[i-1,0], road_center[i,1]-road_center[i-1,1]
        else:
            dx, dy = road_center[i+1,0]-road_center[i,0], road_center[i+1,1]-road_center[i,1]
        norm = np.hypot(dx, dy)
        nx[i], ny[i] = -dy/norm, dx/norm

    planner = FrenetQP_MultiModePlanner()
    
    # 模式定义 (换成标准的中文图例)
    modes = [
        {'id': 'center', 'name': '居中行驶 (基准)', 'color': 'black', 'style': '-'},
        {'id': 'offset', 'name': '固定偏移 (+0.3m)', 'color': 'blue', 'style': '--'},
        {'id': 'out_in_out', 'name': '外-内-外切弯', 'color': 'red', 'style': '-.'}
    ]
    
    results = {}
    for m in modes:
        d_opt = planner.plan(road_s, curve_flags, mode=m['id'])
        x_opt = road_center[:, 0] + d_opt * nx
        y_opt = road_center[:, 1] + d_opt * ny
        v_opt = calc_speed_profile(x_opt, y_opt, road_s, target_speed_kmh=60.0, max_lat_acc=2.5)
        results[m['id']] = {'d': d_opt, 'x': x_opt, 'y': y_opt, 'v': v_opt}

    curve_s = road_s[curve_flags]

    # ---------------------------------------------------------
    # 图 (a)：全局 X-Y 坐标系轨迹图
    # ---------------------------------------------------------
    fig_a = plt.figure(figsize=(8, 6))
    ax_a = fig_a.add_subplot(111)
    
    ax_a.plot(road[:, 0], road[:, 1], 'gray', linewidth=2, label='道路边界')
    ax_a.plot(road[:, 2], road[:, 3], 'gray', linewidth=2)
    ax_a.plot(road[:, 4], road[:, 5], 'gray', linestyle=':', linewidth=1.5, label='中心线')
    
    for m in modes:
        ax_a.plot(results[m['id']]['x'], results[m['id']]['y'], 
                 color=m['color'], linestyle=m['style'], linewidth=2.5, label=m['name'])
    
    ax_a.set_title('(a) 全局坐标系下的多模态规划轨迹', fontsize=14, fontweight='bold', pad=15)
    ax_a.set_xlabel('全局 $\mathrm{X\ [m]}$', fontsize=12)
    ax_a.set_ylabel('全局 Y [m]', fontsize=12)
    ax_a.legend(loc='best', fontsize=11)
    ax_a.grid(True, linestyle='--', alpha=0.6)
    
    # 核心修复：使用 axis('equal') 后，设定相对宽松的限制框，防止切边
    ax_a.axis('equal') 
    ax_a.set_xlim(380, 580)
    ax_a.set_ylim(-370, -220) 
    
    plt.tight_layout()
    fig_a.savefig('fig_a_global_trajectory.png', dpi=600, bbox_inches='tight')
    print("Saved: fig_a_global_trajectory.png")

    # ---------------------------------------------------------
    # 图 (b)：Frenet s-d 横向偏移图
    # ---------------------------------------------------------
    fig_b = plt.figure(figsize=(7, 5))
    ax_b = fig_b.add_subplot(111)
    
    if len(curve_s) > 0:
        ax_b.axvspan(curve_s[0], curve_s[-1], color='yellow', alpha=0.15, label='主弯道区域')
        
    for m in modes:
        ax_b.plot(road_s, results[m['id']]['d'], color=m['color'], linestyle=m['style'], linewidth=2.5, label=m['name'])
        
    ax_b.set_title('(b) $Frenet$坐标系下的横向偏离规划', fontsize=14, fontweight='bold', pad=15)
    ax_b.set_xlabel('纵向弧长 $s$ $\mathrm{[m]}$', fontsize=12)
    ax_b.set_ylabel('横向偏移 $d(s)$ $\mathrm{[m]}$', fontsize=12)
    ax_b.legend(loc='upper left', fontsize=11)
    ax_b.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig_b.savefig('fig_b_frenet_offset.png', dpi=600, bbox_inches='tight')
    print("Saved: fig_b_frenet_offset.png")

    # ---------------------------------------------------------
    # 图 (c)：Frenet s-v 速度曲线图
    # ---------------------------------------------------------
    fig_c = plt.figure(figsize=(7, 5))
    ax_c = fig_c.add_subplot(111)
    
    if len(curve_s) > 0:
        ax_c.axvspan(curve_s[0], curve_s[-1], color='yellow', alpha=0.15, label='主弯道区域')
        
    for m in modes:
        ax_c.plot(road_s, results[m['id']]['v'], color=m['color'], linestyle=m['style'], linewidth=2.5, label=m['name'])
        
    ax_c.set_title('(c) 考虑运动学极值 ($a_{lat} \leq 2.5 m/s^2$) 的理论安全车速', fontsize=14, pad=15)
    ax_c.set_xlabel('纵向弧长 $s$ [m]', fontsize=12)
    ax_c.set_ylabel('纵向车速 $v(s)$ $\mathrm{[km/h]}$', fontsize=12)
    ax_c.set_ylim(20, 65)
    ax_c.legend(loc='lower right', fontsize=11)
    ax_c.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig_c.savefig('fig_c_speed_profile.png', dpi=600, bbox_inches='tight')
    print("Saved: fig_c_speed_profile.png")

    plt.show()