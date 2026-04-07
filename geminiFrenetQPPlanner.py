import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.sparse as sparse
import osqp

# ==============================================================================
# 全局图表样式配置 (博士论文制图强迫症级配置)
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['xtick.direction'] = 'in'     
plt.rcParams['ytick.direction'] = 'in'     
plt.rcParams['font.size'] = 12             
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

# ==============================================================================
# 1. 环境加载与道路解析
# ==============================================================================
class natural_road_load():
    def __init__(self):
        self.s_total = 190.0
        self.s_straight_1 = 30.0   
        self.s_clothoid_in = 40.0   
        self.s_circular_curve = 50.0 
        self.s_clothoid_out = 40.0  
        self.s_straight_2 = 30.0    

    def curve_in_check(self, s):
        # 扩大一点弯道影响区，保证平滑过渡
        if self.s_straight_1 - 10.0 <= s <= self.s_total - self.s_straight_2 + 10.0:
            return True
        return False

# ==============================================================================
# 2. Frenet QP 多模态规划器 (升级：支持早/晚切弯轨迹逻辑)
# ==============================================================================
class FrenetQP_ApexPlanner:
    def __init__(self):
        self.W_dl = 10.0            # 一阶导惩罚 (航向)
        self.W_ddl = 800.0          # 二阶导惩罚 (曲率/舒适度) 
        self.W_dddl = 200.0         # 三阶导惩罚 (Jerk)

    def generate_derivative_matrices(self, N):
        L1 = np.zeros((N-1, N))
        for i in range(N-1): L1[i, i], L1[i, i+1] = -1.0, 1.0
        L2 = np.zeros((N-2, N))
        for i in range(N-2): L2[i, i], L2[i, i+1], L2[i, i+2] = 1.0, -2.0, 1.0
        L3 = np.zeros((N-3, N))
        for i in range(N-3): L3[i, i], L3[i, i+1], L3[i, i+2], L3[i, i+3] = -1.0, 3.0, -3.0, 1.0
        return L1, L2, L3

    def plan(self, road_s, is_in_curve_flags, mode='out_in_out'):
        N = len(road_s)
        W_ref = np.ones(N) * 0.0  
        D_ref = np.zeros(N)
        
        curve_indices = np.where(is_in_curve_flags)[0]
        actual_apex_idx = None # 用于记录实际设置的弯心位置以便返回画图
        
        if len(curve_indices) > 0:
            idx_start, idx_end = curve_indices[0], curve_indices[-1]
            idx_geom_apex = (idx_start + idx_end) // 2 # 几何中点
            
            # 放开弯道内的刚性追踪
            W_ref[idx_start:idx_end+1] = 0.0 
            
            # 统一定义内外侧的横向目标值
            d_out = -0.5  # 外抛深度
            d_in = 0.6    # 切弯深度
            
            # ---------------------------------------------------------
            # 核心逻辑：通过平移 idx_apex 来实现不同的赛道走线
            # ---------------------------------------------------------
            if mode == 'out_in_out':
                # 几何最优线：弯心位于几何中点
                actual_apex_idx = idx_geom_apex
                W_ref[idx_start], D_ref[idx_start] = 500.0, d_out
                W_ref[actual_apex_idx], D_ref[actual_apex_idx] = 1000.0, d_in
                W_ref[idx_end], D_ref[idx_end] = 500.0, d_out
                
            elif mode == 'early_apex':
                # 早切弯：弯心提前 25m
                actual_apex_idx = idx_geom_apex - 25
                W_ref[idx_start], D_ref[idx_start] = 800.0, d_out
                W_ref[actual_apex_idx], D_ref[actual_apex_idx] = 1000.0, d_in
                # 早切弯往往导致出弯推头外抛更严重，设定一个更靠外的目标
                W_ref[idx_end], D_ref[idx_end] = 500.0, -0.7 
                
            elif mode == 'late_apex':
                # 晚切弯：弯心延后 25m
                actual_apex_idx = idx_geom_apex + 25
                # 入弯在外侧保持更久，可以加重入弯点的约束
                W_ref[idx_start], D_ref[idx_start] = 1000.0, d_out
                W_ref[actual_apex_idx], D_ref[actual_apex_idx] = 1000.0, d_in
                W_ref[idx_end], D_ref[idx_end] = 500.0, d_out 

        # 构造求解矩阵
        L1, L2, L3 = self.generate_derivative_matrices(N)
        H = 2.0 * np.diag(W_ref) + \
            2.0 * self.W_dl * (L1.T @ L1) + \
            2.0 * self.W_ddl * (L2.T @ L2) + \
            2.0 * self.W_dddl * (L3.T @ L3)
        P = sparse.csc_matrix(H)
        q = -2.0 * (W_ref * D_ref)

        A_d = sparse.eye(N, format='csc')
        l_d = np.ones(N) * -0.8
        u_d = np.ones(N) * 0.8
        
        if len(curve_indices) > 0:
            l_d[:idx_start-5] = 0.0
            u_d[:idx_start-5] = 0.0
            l_d[idx_end+5:] = 0.0
            u_d[idx_end+5:] = 0.0

        A_dl = sparse.csc_matrix(L1)
        l_dl = np.ones(N-1) * -0.5
        u_dl = np.ones(N-1) * 0.5

        A_ddl = sparse.csc_matrix(L2)
        l_ddl = np.ones(N-2) * -0.05
        u_ddl = np.ones(N-2) * 0.05

        A = sparse.vstack([A_d, A_dl, A_ddl], format='csc')
        l = np.hstack([l_d, l_dl, l_ddl])
        u = np.hstack([u_d, u_dl, u_ddl])

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
        
        res = prob.solve()
        if res.info.status_val != 1:
            print(f"QP Solver Failed for {mode}!")
            return np.zeros(N), actual_apex_idx

        return res.x, actual_apex_idx

# ==============================================================================
# 3. 数据生成与制图 
# ==============================================================================
if __name__ == '__main__':
    env = natural_road_load()
    road_s = np.arange(0, env.s_total + 1.0, 1.0)
    curve_flags = np.array([env.curve_in_check(s) for s in road_s])
    
    s_roi_start = 20.0
    s_roi_end = 170.0

    planner = FrenetQP_ApexPlanner()
    
    fig = plt.figure(figsize=(12, 7)) 
    ax = fig.add_subplot(111)

    # 绘制基础车道线
    ax.plot(road_s, np.zeros_like(road_s), 'k-', linewidth=1.0, alpha=0.5, label='车道中心线')
    ax.plot(road_s, np.ones_like(road_s)*0.8, 'k--', linewidth=1.5, alpha=0.5, label='安全边界')
    ax.plot(road_s, np.ones_like(road_s)*(-0.8), 'k--', linewidth=1.5, alpha=0.5)
    
    # 定义需要生成的三种轨迹配置
    modes = [
        {'id': 'out_in_out', 'name': '外内外 (几何最优)', 'color': '#27AE60'}, # 绿色
        {'id': 'early_apex', 'name': '早切弯 (Early Apex)', 'color': '#C0392B'}, # 红色
        {'id': 'late_apex',  'name': '晚切弯 (Late Apex)',  'color': '#2980B9'}  # 蓝色
    ]
    
    for m in modes:
        d_opt, apex_idx = planner.plan(road_s, curve_flags, mode=m['id'])
        
        # 绘制整条轨迹线
        ax.plot(road_s, d_opt, color=m['color'], linestyle='-', linewidth=2.5, label=m['name'])
        
        # 在轨迹上用醒目的五角星标出 "弯心 (Apex)" 的位置
        if apex_idx is not None:
            ax.plot(road_s[apex_idx], d_opt[apex_idx], marker='*', color=m['color'], 
                    markersize=15, markeredgecolor='black', markeredgewidth=0.5)
    
    # ROI 分界线
    ax.axvline(s_roi_start, color='gray', linestyle='-.', linewidth=1.2)
    ax.axvline(s_roi_end, color='gray', linestyle='-.', linewidth=1.2)
    
    # 标签与排版
    ax.set_xlabel("纵向距离 $\mathrm{s/m}$", fontsize=20, labelpad=10)
    ax.set_ylabel("横向偏移量 $\mathrm{d/m}$", fontsize=20, labelpad=10)
    ax.set_xlim(road_s[0], road_s[-1]) 
    ax.set_ylim(-1.0, 1.0) 

    # 优化图例排版，去掉边框
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, 
              fontsize=16, frameon=False)
    
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()