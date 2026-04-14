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
plt.rcParams['font.size'] = 14             # 稍微调大了一点字号，图例更清晰
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
# 2. Frenet QP 多模态规划器 (完美复现自然驾驶聚类特征)
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

    def plan(self, road_s, is_in_curve_flags, mode='cluster_1'):
        N = len(road_s)
        W_ref = np.ones(N) * 0.0  
        D_ref = np.zeros(N)
        
        curve_indices = np.where(is_in_curve_flags)[0]
        actual_apex_idx = None # 用于记录实际设置的特征点位置以便返回画图
        
        if len(curve_indices) > 0:
            idx_start, idx_end = curve_indices[0], curve_indices[-1]
            idx_geom_apex = (idx_start + idx_end) // 2 # 几何中点
            
            # 放开弯道内的刚性追踪
            W_ref[idx_start:idx_end+1] = 0.0 
            
            # ---------------------------------------------------------
            # 核心参数整定：复现右转弯道的 4 类驾驶行为
            # (注意：左侧为正 d>0，右侧内弯为负 d<0)
            # ---------------------------------------------------------
            if mode == 'cluster_1':
                # 第一类：深切弯与内侧贴合 (Aggressive Inner)
                actual_apex_idx = idx_geom_apex + 10 # 极值点稍微靠后
                W_ref[idx_start], D_ref[idx_start] = 300.0, 0.05
                W_ref[actual_apex_idx], D_ref[actual_apex_idx] = 1000.0, -0.45 # 深切弯心
                W_ref[idx_end], D_ref[idx_end] = 800.0, -0.25 # 出弯保持在内侧，慵懒回正
                
            elif mode == 'cluster_2':
                # 第二类：保守中心跟随 (Conservative Center)
                actual_apex_idx = idx_geom_apex
                W_ref[idx_start], D_ref[idx_start] = 800.0, 0.0
                W_ref[actual_apex_idx], D_ref[actual_apex_idx] = 800.0, 0.05 # 弯心略微偏外侧
                W_ref[idx_end], D_ref[idx_end] = 800.0, 0.0
                
            elif mode == 'cluster_3':
                # 第三类：弯中外抛 (Outer-Deviating)
                actual_apex_idx = idx_geom_apex + 5
                W_ref[idx_start], D_ref[idx_start] = 500.0, 0.0
                W_ref[actual_apex_idx], D_ref[actual_apex_idx] = 800.0, 0.35 # 弯心出现明显的正值外抛
                W_ref[idx_end], D_ref[idx_end] = 500.0, 0.0
                
            elif mode == 'cluster_4':
                # 第四类：典型外内外与早切弯 (Early Apex Out-In-Out)
                actual_apex_idx = idx_geom_apex - 15 # 特征点：明显的早弯心
                W_ref[idx_start], D_ref[idx_start] = 500.0, 0.25 # 借外侧空间入弯
                W_ref[actual_apex_idx], D_ref[actual_apex_idx] = 1000.0, -0.4 # 极早切入内侧
                W_ref[idx_end], D_ref[idx_end] = 500.0, 0.3 # 出弯向外侧甩出

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
        
        # [修改点] 为了让第一类和第四类在进出弯有足够的距离平滑过渡，
        # 将直线居中约束的生效位置拉远到弯道边界外 20 米处。
        if len(curve_indices) > 0:
            l_d[:max(0, idx_start-20)] = 0.0
            u_d[:max(0, idx_start-20)] = 0.0
            l_d[min(N, idx_end+20):] = 0.0
            u_d[min(N, idx_end+20):] = 0.0

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
    ax.plot(road_s, np.zeros_like(road_s), 'k-', linewidth=1.5, alpha=0.5, label='车道中心线')
    ax.plot(road_s, np.ones_like(road_s)*0.8, 'k--', linewidth=1.5, alpha=0.5, label='安全边界')
    ax.plot(road_s, np.ones_like(road_s)*(-0.8), 'k--', linewidth=1.5, alpha=0.5)
    
    # 重新定义的 4 种聚类轨迹配置
    modes = [
        {'id': 'cluster_1', 'name': '(a) 第一类：深切弯', 'color': '#27AE60'}, # 绿色
        {'id': 'cluster_2', 'name': '(b) 第二类：中心跟随', 'color': '#2980B9'}, # 蓝色
        {'id': 'cluster_3', 'name': '(c) 第三类：弯中外抛', 'color': '#F39C12'}, # 橙色
        {'id': 'cluster_4', 'name': '(d) 第四类：早切与外内外', 'color': '#C0392B'}  # 红色
    ]
    
    for m in modes:
        d_opt, apex_idx = planner.plan(road_s, curve_flags, mode=m['id'])
        
        # 绘制整条轨迹线
        ax.plot(road_s, d_opt, color=m['color'], linestyle='-', linewidth=3.0, label=m['name'])
        
        # 在轨迹上用醒目的五角星标出极值点/控制点的位置
        if apex_idx is not None:
            ax.plot(road_s[apex_idx], d_opt[apex_idx], marker='*', color=m['color'], 
                    markersize=18, markeredgecolor='black', markeredgewidth=1.0)
    
    # ROI 分界线
    ax.axvline(s_roi_start, color='gray', linestyle='-.', linewidth=1.2)
    ax.axvline(s_roi_end, color='gray', linestyle='-.', linewidth=1.2)
    
    # 标签与排版
    ax.set_xlabel("纵向距离 $s \mathrm{/m}$", fontsize=20, labelpad=10)
    ax.set_ylabel("横向偏移量 $d \mathrm{/m}$", fontsize=20, labelpad=10)
    ax.set_xlim(road_s[0], road_s[-1]) 
    ax.set_ylim(-1.0, 1.0) 

    # 采用 2x2 或 1x4 均可，这里使用 1x4 放最上面
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, 
              fontsize=16, frameon=False)
    
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    # fig.savefig('four_clusters_qp_simulation.png', dpi=600, bbox_inches='tight')
    plt.show()