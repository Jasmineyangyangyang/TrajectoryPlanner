# import numpy as np
# import matplotlib.pyplot as plt
# import copy
# import math
# import traject_plan_control.cubic_spline_planner as cubic_spline_planner

# for debug
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import math
import sys
import os
import pathlib
current_dir = pathlib.Path(os.getcwd())
for subdir in current_dir.iterdir():
    if subdir.is_dir():
        sys.path.append(str(subdir))
from CubicSpline import cubic_spline_planner
from global_road import natural_road_load
from scipy.ndimage import gaussian_filter1d 
import time

"""
单车道宽度：3.75m
车宽：1.85m
车辆允许的单侧max offset = 3.75/2 - 1.85/2 = 0.95m
"""

# Parameter
MAX_SPEED = 40.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.5  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
#----#
road_width = 3.75 #m
vehicle_width = 1.85 #m
offset_buffer = 0.15 #m, buffer for offset
MAX_ROAD_WIDTH = round(road_width/2 - vehicle_width/2 - offset_buffer, 2)  # maximum road width [m]，保留两位小数
D_ROAD_W = 0.2  # road width sampling length [m]
DT = 0.3  # firts searching time tick [s]
# DT_best_path = 0.1  # best path searching time tick [s]
DT_best_path = 0.01  # best path searching time tick [s]
PLAN_T = 5.0  # max prediction time [m]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 2  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

# init cost weights
K_J = 0.0
K_D = 0.0
K_T = 1.0 - K_J - K_D

# lateral bias configuration (used inside calc_frenet_paths cost)
# - "curve_in_out_by_KT": use reference curvature sign to define inside/outside of a curve
#     inside_metric = d_end * sign(kappa_ref). Minimizing a coefficient * inside_metric will prefer:
#       - K_T small  -> coefficient (1-2*K_T) > 0 -> inside_metric negative -> outside
#       - K_T large  -> coefficient (1-2*K_T) < 0 -> inside_metric positive -> inside
# - "left_right_by_KT": treat d>0 as "left" and d<0 as "right" (independent of curvature)
LATERAL_BIAS_MODE = "left_right_by_KT"
LATERAL_BIAS_GAIN = 1.0
CURVATURE_SIGN_EPS = 1e-6

def _as_scalar(x):
    """
    Ensure numeric inputs are Python floats.
    Accepts Python numbers or numpy scalars/size-1 arrays.
    """
    x_arr = np.asarray(x)
    if x_arr.size != 1:
        raise ValueError(f"Expected scalar, got shape={x_arr.shape}")
    return float(x_arr.reshape(()))

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, time, axe=0.0):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, time, vxe=0.0, axe=0.0):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt

class FrenetPath:
    def __init__(self):
        self.t = []     # time
        self.l = []     # lateral position
        self.l_dot = []   # lateral speed
        self.l_ddot = []  # lateral acceleration
        self.l_dddot = [] # lateral jerk
        self.s = []     # longitudinal position
        self.s_dot = []   # longitudinal speed
        self.s_ddot = []  # longitudinal acceleration
        self.s_dddot = [] # longitudinal jerk
        self.cd = 0.0   # cost d
        self.cv = 0.0   # cost v
        self.cf = 0.0   # cost sum

        self.x = []     # global x position
        self.y = []     # global y position
        self.yaw = []   # global yaw position
        self.speed = [] # global speed
        self.a = []     # global acceleration
        self.c = []     # global curvature

        self.lat_param = []  # lateral motion planning parameter
        self.lon_param = []  # longitudinal motion planning parameter

def calc_frenet_path(lat_param, lon_param):
    fp = FrenetPath()
    lat_qp = QuinticPolynomial(lat_param[0], lat_param[1], lat_param[2], lat_param[3], lat_param[4])
    lon_qp = QuarticPolynomial(lon_param[0], lon_param[1], lon_param[2], lon_param[3], lon_param[4])
    
    fp.t = [t for t in np.arange(0.0, PLAN_T, DT_best_path)]

    fp.l = [lat_qp.calc_point(t) for t in fp.t]
    fp.l_dot = [lat_qp.calc_first_derivative(t) for t in fp.t]
    fp.l_ddot = [lat_qp.calc_second_derivative(t) for t in fp.t]
    fp.l_dddot = [lat_qp.calc_third_derivative(t) for t in fp.t]

    fp.s = [lon_qp.calc_point(t) for t in fp.t]
    fp.s_dot = [lon_qp.calc_first_derivative(t) for t in fp.t]
    fp.s_ddot = [lon_qp.calc_second_derivative(t) for t in fp.t]
    fp.s_dddot = [lon_qp.calc_third_derivative(t) for t in fp.t]

    return fp
    
def calc_frenet_paths(csp, s0, s0_dot, s0_ddot, l0, l0_dot, l0_ddot, planner_param, TARGET_SPEED):
    """# target speed [m/s]"""
    MAX_LON_ACC = 3.0   # 或者 2.5
    MAX_LON_JERK = 2.5
    MAX_LAT_ACC = 0.5   # 贴合你跑出来的 0.3 上限，稍微留点裕度
    MAX_LAT_JERK = 1.0  # 贴合你跑出来的 1.0 上限
    max_speed_error = 2 * D_T_S * N_S_SAMPLE  # maximum possible speed error based on sampling range


    frenet_paths = []
    test_cost = []

    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH+D_ROAD_W, D_ROAD_W):
    # Lateral motion planning
        Ti = PLAN_T   # acctually you can search different Ti, but that will cost more time
        fp = FrenetPath()

        lat_qp = QuinticPolynomial(l0, l0_dot, l0_ddot, di, Ti)  # 低速时将Ti理解成Si

        fp.t = [t for t in np.arange(0.0, Ti+DT, DT)]
        fp.l = [lat_qp.calc_point(t) for t in fp.t]
        fp.l_dot = [lat_qp.calc_first_derivative(t) for t in fp.t]
        fp.l_ddot = [lat_qp.calc_second_derivative(t) for t in fp.t]
        fp.l_dddot = [lat_qp.calc_third_derivative(t) for t in fp.t]
        fp.lat_param = [l0, l0_dot, l0_ddot, di, Ti]

        # Longitudinal motion planning (Velocity keeping)
        for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                            TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
            tfp = copy.deepcopy(fp)

            lon_qp = QuarticPolynomial(s0, s0_dot, s0_ddot, tv, Ti)

            tfp.s = [lon_qp.calc_point(t) for t in fp.t]
            tfp.s_dot = [lon_qp.calc_first_derivative(t) for t in fp.t]
            tfp.s_ddot = [lon_qp.calc_second_derivative(t) for t in fp.t]
            tfp.s_dddot = [lon_qp.calc_third_derivative(t) for t in fp.t]
            tfp.lon_param = [s0, s0_dot, s0_ddot, tv, Ti]

            # =====================================
            # 将权重限制在 [0,1] 且三者和为 1，方便直接试 [0,1] 组合
            # =====================================
            KJ = max(0.0, min(1.0, planner_param[0]))  # jerk 权重
            KD = max(0.0, min(1.0, planner_param[1]))  # 终点偏移 / 速度误差 权重
            sum_w = KJ + KD
            if sum_w > 1.0:
                # 如果前两项之和超过 1，则按比例缩放到和为 1
                KJ /= sum_w
                KD /= sum_w
            KT  = max(0.0, 1.0 - KJ - KD)  # 剩余权重给效率

            # =================================================
            # raw cost
            # =================================================
            print(f"max l_dddot: {max(tfp.l_dddot)}, max s_dddot: {max(tfp.s_dddot)}")
            print(f"max l_ddot: {max(tfp.l_ddot)}, max s_ddot: {max(tfp.s_ddot)}")
            raw_lat_jerk = sum(np.power(tfp.l_dddot, 2)) * DT  # square of lat jerk
            raw_lon_jerk = sum(np.power(tfp.s_dddot, 2)) * DT  # square of lon jerk

            # efficiency cost
            raw_speed_error = (TARGET_SPEED - tfp.s_dot[-1]) ** 2
            raw_acc_lon = sum(np.power(tfp.s_ddot, 2)) * DT  # square of acceleration error
            raw_acc_lat = sum(np.power(tfp.l_ddot, 2)) * DT  # square of lat acc

            # offset from lane center at the end of prediction horizon
            # l_ref = -MAX_ROAD_WIDTH + 2.0 * MAX_ROAD_WIDTH * KD
            l_ref = -MAX_ROAD_WIDTH + 2.0 * MAX_ROAD_WIDTH * (1-KD)
            raw_bias_error = abs(tfp.l[-1] - l_ref)

            # progress reward: distance along the path at the end of prediction horizon
            progress_reward = tfp.s[-1]

            # =================================================
            # physical normalization
            # =================================================
            norm_jerk_cost = raw_lat_jerk / (MAX_LAT_JERK ** 2 * Ti) + raw_lon_jerk / (MAX_LON_JERK ** 2 * Ti)  # normalized jerk cost
            norm_speed_cost = raw_speed_error / (max_speed_error ** 2)  # normalized speed cost
            norm_acc_cost = raw_acc_lon / (MAX_LON_ACC ** 2 * Ti) + raw_acc_lat / (MAX_LAT_ACC ** 2 * Ti)  # normalized acceleration cost
            norm_bias_cost = raw_bias_error / (2*MAX_ROAD_WIDTH)  # normalized offset cost
            norm_progress_cost = -progress_reward / (TARGET_SPEED * Ti)  # normalized progress cost


            # =====================================
            # final weighted cost
            # =====================================
            # 将权重限制在 [0,1] 且三者和为 1，方便直接试 [0,1] 组合
            

            tfp.cd = norm_jerk_cost
            tfp.cv = norm_speed_cost + norm_acc_cost + norm_progress_cost
            tfp.cf = KJ * tfp.cd + norm_bias_cost + KT * tfp.cv

            frenet_paths.append(tfp)
            test_cost.append(tfp.cd)

    return frenet_paths

def check_collision(fp, ob):
    if len(ob) == 0:
        return True
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True

def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].speed]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].a]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    # return [fplist[i] for i in ok_ind]
    return ok_ind

def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))
    continuous_ryaw = np.unwrap(np.array(ryaw))
    ryaw = continuous_ryaw.tolist()
    rs = s.tolist()
    return rx, ry, ryaw, rk, rs, csp

def curvature_from_yaw_rate(yaw_rate, speed, min_speed=0.1):
    """
    由横摆角速度与车速计算当前轨迹曲率：κ = yaw_rate / v。
    真实车或仿真有 yaw_rate 时，可用此函数得到 kappa，再传给 cartesian_to_frenet_state
    或 poly_trajectory(..., ego_kappa=...)，以提升弯道下 Frenet 加速度 s̈、l̈ 的精度。

    Args:
        yaw_rate: 横摆角速度 [rad/s]
        speed: 车速 [m/s]
        min_speed: 速度过小时的分母下限，避免除零或数值爆炸 [m/s]

    Returns:
        轨迹曲率 κ [1/m]
    """
    v = max(abs(speed), min_speed)
    return float(yaw_rate) / v

class Polyplanner():
    def __init__(self, env_data, lane_id):

        self.env_data = env_data
        # self.road = self.env_data.read_from_csv('./trajectplanner')
        self.road = self.env_data.read_from_csv('./')
        self.road_center = []   # this is the Cartesian center of the lane which ego vehicle is driving.
        self.road_left = self.road[:,0:2]     # [x, y]
        self.road_right = self.road[:,2:4]    # [x, y]
        for i in range(len(self.road)):
            if lane_id == 0:     # outside
                center_left_x = self.road[i][0]
                center_left_y = self.road[i][1]
                center_right_x = self.road[i][4]
                center_right_y = self.road[i][5]
                
            elif lane_id == 1:   # inside
                center_left_x = self.road[i][4]
                center_left_y = self.road[i][5]
                center_right_x = self.road[i][2]
                center_right_y = self.road[i][3]

            road_center_x = (center_left_x + center_right_x) / 2.0
            road_center_y = (center_left_y + center_right_y) / 2.0
            self.road_center.append([road_center_x, road_center_y])
        self.road_center = np.array(self.road_center)  # Cartesian coordinate
        self.wx = self.road_center[:,0]
        self.wy = self.road_center[:,1]
        self.wx =np.append(self.wx, self.wx[0])
        self.wy = np.append(self.wy, self.wy[0])
        self.tx, self.ty, self.tyaw, self.tc, self.ts, self.csp = generate_target_course(self.wx, self.wy)

        # 参考路径（用于全状态 Frenet 转换）
        self.ref_x = np.array(self.tx)
        self.ref_y = np.array(self.ty)
        self.ref_psi = np.unwrap(np.array(self.tyaw))

        # 以弧长为自变量构造 s 序列，并据此计算 κ' = dκ/ds
        self.ref_s = np.array(self.ts)
        self.ref_kappa = np.array(self.tc)
        # 为避免数值噪声，这里用 np.gradient 近似 dκ/ds
        self.ref_dkappa_ds = np.gradient(self.ref_kappa, self.ref_s)
        kappa_smoothed = gaussian_filter1d(self.ref_kappa, sigma=30)
        self.ref_kappa = kappa_smoothed
        self.ref_dkappa_ds = np.gradient(kappa_smoothed, self.ref_s)
        self.show_animation = True
    
    def find_nearest_point(self, x, y):
        # 计算每个参考点到 (x, y) 的距离
        distances = np.sqrt((np.array(self.tx) - x)**2 + (np.array(self.ty) - y)**2)
        # 找到最近的参考点的索引
        nearest_index = np.argmin(distances)
        
        return nearest_index

    def cartesian_to_frenet_state(self, x, y, theta, v, a=0.0, kappa=0.0):
        """
        优化后的笛卡尔到Frenet坐标转换。
        通过最近点附近的线性插值/局部搜索寻找精确投影位置。
        """
        # ---- Type unification: force all inputs to scalar floats ----
        x = _as_scalar(x)
        y = _as_scalar(y)
        theta = _as_scalar(theta)
        v = _as_scalar(v)
        a = _as_scalar(a)
        kappa = _as_scalar(kappa)

        # 1. 粗寻：找到最近的离散参考点索引
        distances = np.sqrt((self.ref_x - x)**2 + (self.ref_y - y)**2)
        idx = np.argmin(distances)

        # # 2. 精寻：在最近点前后利用几何关系逼近真实投影点 s, 【精度为1e-3不够】
        # # 设投影点为 P，当前车位为 M，最近点为 R_idx，前一点或后一点为 R_adj
        # if idx == 0:
        #     idx_next = 1
        # elif idx == len(self.ref_s) - 1:
        #     idx_next = idx - 1
        # else:
        #     # 判断投影点是在 idx 的前面还是后面
        #     d_prev = distances[idx-1]
        #     d_next = distances[idx+1]
        #     idx_next = idx + 1 if d_next < d_prev else idx - 1

        # # 计算 R_idx 到 R_next 的向量
        # dx_r = self.ref_x[idx_next] - self.ref_x[idx]
        # dy_r = self.ref_y[idx_next] - self.ref_y[idx]
        # ds_r = self.ref_s[idx_next] - self.ref_s[idx]

        # # 向量 RM (从最近参考点指向车辆实际位置)
        # dx_m = x - self.ref_x[idx]
        # dy_m = y - self.ref_y[idx]
        
        # # 将 RM 投影到道路切向向量上，计算相对于 s_idx 的偏移量
        # # offset = (RM · unit_vector_of_road)
        # r_mag = math.hypot(dx_r, dy_r)
        # if r_mag > 1e-6:
        #     s_offset = (dx_m * dx_r + dy_m * dy_r) / r_mag
        #     s = (self.ref_s[idx] + s_offset) % self.ref_s[-1]
        # else:
        #     s = self.ref_s[idx]
        if idx == 0:
            s_guess = self.ref_s[idx] + 1e-5
        else:
            s_guess = self.ref_s[idx]

        # 2. 精寻：牛顿迭代修正 s (1-2次迭代即可达到极高精度) 【精度达到1e-4】
        # 核心思想：s = s + (P - Pr)·Tr，即在切向上的投影误差
        for _ in range(2):
            xr, yr = self.csp.calc_position(s_guess)
            psi_r = self.csp.calc_yaw(s_guess)
            dx = x - xr
            dy = y - yr
            # 计算切向投影量
            s_error = dx * math.cos(psi_r) + dy * math.sin(psi_r)
            s_guess = max(s_guess + s_error, 1e-5)
        
        s = s_guess % self.ref_s[-1]

        # 3. 基于精确 s 获取参考点属性 (使用样条插值)
        # 获取精确的参考点坐标、航向角、曲率及其对 s 的导数
        xr, yr = self.csp.calc_position(s)
        psi_r = self.csp.calc_yaw(s)
        # psi_r = float(np.interp(s, self.ref_s, self.ref_psi))
        s_mod = s % self.ref_s[-1]
        kappa_r = self.csp.calc_curvature(s_mod)
        if kappa_r is None:
            kappa_r = 0.0
        kappa_r = float(kappa_r)
        
        # # 利用高斯平滑后的曲率计算 kappa_prime 【导致s_ddot误差约1.5，太大】
        # kappa_rp = float(np.interp(s_mod, self.ref_s, self.ref_dkappa_ds))

        # 局部数值法计算 kappa_prime (d_kappa/ds)
        # 避开平滑数组，确保 kappa_rp 是当前 kappa_r 的真实导数
        ds = 1e-3
        k_plus = self.csp.calc_curvature((s + ds) % self.ref_s[-1])
        k_minus = self.csp.calc_curvature((s - ds) % self.ref_s[-1])
        k_plus = 0.0 if k_plus is None else float(k_plus)
        k_minus = 0.0 if k_minus is None else float(k_minus)
        kappa_rp = (k_plus - k_minus) / (2.0 * ds)

        # 4. 计算 Frenet 状态量
        t_r = np.array([math.cos(psi_r), math.sin(psi_r)])
        n_r = np.array([-math.sin(psi_r), math.cos(psi_r)])
        v_rel = np.array([x - float(xr), y - float(yr)])

        l = float(np.dot(v_rel, n_r))  # 精确的侧向距离（标量）
        
        # 这里的 q1 = 1 - kappa_r * l 是转换的核心系数
        q1 = 1.0 - kappa_r * l
        if abs(q1) < 1e-3:
            q1 = np.sign(q1)*1e-3 # 防止在极小转弯半径下出现奇点
        # 航向角偏差
        delta_psi = math.atan2(math.sin(theta - float(psi_r)), math.cos(theta - float(psi_r)))
        
        # 速度分量转换
        s_dot = v * math.cos(delta_psi) / q1
        l_dot = v * math.sin(delta_psi)

        # 加速度向量投影
        t_c = np.array([math.cos(theta), math.sin(theta)])
        n_c = np.array([-math.sin(theta), math.cos(theta)])
        a_vec = a * t_c + (v ** 2) * kappa * n_c  # 考虑当前曲率 kappa 带来的向心加速度
        
        a_t_r = float(np.dot(a_vec, t_r))
        a_n_r = float(np.dot(a_vec, n_r))

        # 计算 s_ddot 和 l_ddot
        s_ddot = (a_t_r + kappa_rp * l * s_dot ** 2 + 2.0 * kappa_r * s_dot * l_dot) / q1
        l_ddot = a_n_r - q1 * kappa_r * s_dot ** 2

        return float(s), float(l), float(s_dot), float(l_dot), float(s_ddot), float(l_ddot)

    def calculate_frenet_coordinates(self, x, y, yaw, speed, kappa=0.0, a=0.0):
        """
        封装 cartesian_to_frenet_state，返回 (s, s_d, s_dd, d, d_d, d_dd)，
        直接可以作为多项式规划的初始 Frenet 状态。
        有 yaw_rate 时可用：kappa = curvature_from_yaw_rate(yaw_rate, speed)。
        """
        x = _as_scalar(x)
        y = _as_scalar(y)
        yaw = _as_scalar(yaw)
        speed = _as_scalar(speed)
        kappa = _as_scalar(kappa)
        a = _as_scalar(a)

        # 有 yaw_rate 时：kappa = curvature_from_yaw_rate(yaw_rate, speed)，再传入
        s, l, s_dot, l_dot, s_ddot, l_ddot = self.cartesian_to_frenet_state(
            x, y, yaw, speed, a, kappa=kappa
        )
        return float(s), float(s_dot), float(s_ddot), float(l), float(l_dot), float(l_ddot) # s0, c_speed, c_accel, c_d, c_d_d, c_d_dd 

    def calc_global_paths(self, fp, csp):
        # calc global positions
        for i in range(len(fp.s)):
            if fp.s[i] <= 0 or fp.s[i] >= csp.s[-1]:
                ix, iy = csp.calc_position(0)
                i_yaw = csp.calc_yaw(0)
                i_curv = csp.calc_curvature(0)
                ds = 1e-4
                k_plus = csp.calc_curvature((0 + 2 * ds) % self.ref_s[-1])
                k_minus = csp.calc_curvature((0 - ds) % self.ref_s[-1])
                i_curv_prime = (k_plus - k_minus) / (2 * ds)
            else:
                ix, iy = csp.calc_position(fp.s[i])
                i_yaw = csp.calc_yaw(fp.s[i])
                i_curv = csp.calc_curvature(fp.s[i])
                s_mod = fp.s[i] % self.ref_s[-1]
                ds = 1e-4
                k_plus = csp.calc_curvature((s_mod + ds) % self.ref_s[-1])
                k_minus = csp.calc_curvature((s_mod - ds) % self.ref_s[-1])
                i_curv_prime = (k_plus - k_minus) / (2 * ds)

            if ix is None:
                break
            
            di = fp.l[i]
            fx = float(ix - di * math.sin(i_yaw))
            fy = float(iy + di * math.cos(i_yaw))
            fp.x.append(fx)
            fp.y.append(fy)

            # calc speed
            q1 = 1.0 - i_curv * di
            v_t = q1 * fp.s_dot[i]
            v_n = fp.l_dot[i]
            fp.speed.append(np.hypot(v_t, v_n))

            delta_psi = np.arctan2(v_n, v_t)
            fp.yaw.append(i_yaw + delta_psi)

            a_t_r = (
                q1 * fp.s_ddot[i]
                # - i_curv * fp.l[i] * fp.s_dot[i]**2
                - i_curv_prime * di * fp.s_dot[i]**2
                - 2.0 * i_curv * fp.s_dot[i] * fp.l_dot[i]
            )
            a_n_r = q1 * i_curv * fp.s_dot[i]**2 + fp.l_ddot[i]
            v_eps = 1e-6
            v_safe = np.where(fp.speed[i] < v_eps, v_eps, fp.speed[i])

            fp.c.append(float((v_t * a_n_r - v_n * a_t_r) / (v_safe**3)))
            fp.a.append(float((v_t * a_t_r + v_n * a_n_r) / v_safe))

            fp.c[i] = float(np.where(fp.speed[i] < v_eps, 0.0, fp.c[i]))
            fp.a[i] = float(np.where(fp.speed[i] < v_eps, 0.0, fp.a[i]))

        return fp

    def frenet_optimal_planning(self, csp, s0, s0_dot, s0_ddot, l0, l0_dot, l0_ddot, planner_param, target_speed, ob):
        s0 = _as_scalar(s0)
        s0_dot = _as_scalar(s0_dot)
        s0_ddot = _as_scalar(s0_ddot)
        l0 = _as_scalar(l0)
        l0_dot = _as_scalar(l0_dot)
        l0_ddot = _as_scalar(l0_ddot)
        target_speed = _as_scalar(target_speed)

        fplist = calc_frenet_paths(csp, s0, s0_dot, s0_ddot, l0, l0_dot, l0_ddot, planner_param, target_speed)
        # fplist_ok_ind = check_paths(fplist, ob)  # check maximum speed, accel, curvature, collision
        # fplist = [fplist[i] for i in fplist_ok_ind]
        # for i in range(len(fplist)):
        #     plt.plot(fplist[i].s, fplist[i].l, label=f"{fplist[i].cf:.2f}")
        #     plt.legend()
        # find minimum cost path id
        min_cost = float("inf")
        best_path_id = None
        for i in range(len(fplist)):
            if min_cost >= fplist[i].cf:
                min_cost = fplist[i].cf
                best_path_id = i
        # print(f"best_path_id: {best_path_id}")

        frenet_fp = calc_frenet_path(fplist[best_path_id].lat_param, fplist[best_path_id].lon_param)
        fp = self.calc_global_paths(frenet_fp, csp)

        return fp

    def poly_trajectory(self, ego_x, ego_y, ego_speed, planner_param, target_speed,
                        ob=np.array([]), ego_yaw=None, ego_a=0.0, ego_kappa=0.0):
        """
        轨迹规划接口。模拟器/真车通常给 (x, y, speed, a)，无曲率时保持默认即可：
        - ego_x, ego_y, ego_speed, ego_a：可直接提供
        - ego_yaw：车辆航向（若无则传 None，用参考线朝向近似）
        - ego_kappa：车辆曲率。有 yaw_rate 时可用 curvature_from_yaw_rate(ego_yaw_rate, ego_speed)
        """
        ego_x = _as_scalar(ego_x)
        ego_y = _as_scalar(ego_y)
        ego_speed = _as_scalar(ego_speed)
        ego_a = _as_scalar(ego_a)
        ego_kappa = _as_scalar(ego_kappa)
        target_speed = _as_scalar(target_speed)

        if ego_yaw is None:
            nearest_index = self.find_nearest_point(ego_x, ego_y)
            ego_yaw = self.tyaw[nearest_index]
        ego_yaw = _as_scalar(ego_yaw)

        s0, s0_dot, s0_ddot, l0, l0_dot, l0_ddot  = self.calculate_frenet_coordinates(
            ego_x, ego_y, ego_yaw, ego_speed, ego_kappa, ego_a
        )

        path = self.frenet_optimal_planning(self.csp, s0, s0_dot, s0_ddot, l0, l0_dot, l0_ddot,
                                       planner_param, target_speed, ob)
        return path
    
    def debug_sim_frenet_plan_global(self):
        ego_x = self.tx[0]
        ego_y = self.ty[0]
        ego_speed = 40.0 / 3.6  # current speed [m/s]
        ego_yaw = self.tyaw[0]
        ego_kappa = self.tc[0]
        ego_a = 0.0
        target_speed = 60.0 / 3.6
        ob = np.array([])
        # param = [0.0, 0.5]  # center line
        # param = [0.0, 1.0] # innner offset
        param = [0.0, 0.0] # outer offset

        SIM_LOOP = 8000 # simulation loop

        plt.figure(3)
        plt.rcParams['xtick.direction'] = 'in'  #将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'SimSun'],  # 英文字体为新罗马，中文字体为宋体
            'font.sans-serif': ['Times New Roman', 'SimSun'],  # 无衬线字体
            'font.serif': ['Times New Roman', 'SimSun'],  # 衬线字体
            'mathtext.fontset': 'custom', # 设置Latex字体为用户自定义, mathtext的字体是与font.sans-serif绑定的
            'mathtext.default': 'rm',  # 设置mathtext的默认字体为Times New Roman
            # 设置mathtext的无衬线字体为Times New Roman
            'mathtext.sf': 'Times New Roman',  # 设置mathtext的无衬线字体为Times New Roman
            'mathtext.rm': 'Times New Roman',  # 设置mathtext的衬线字体为Times New Roman
            'font.size': 12,  # 五号字
            'axes.unicode_minus': False,  # 解决负号显示问题
            'text.usetex': False,
            'figure.figsize': (3.5, 2.625), #单位是inches, 按IEEE RAL 要求，双栏图片(7.16, 5.37), 单栏图片(3.5, 2.625)
            'figure.dpi': 600,
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'lines.linewidth': 1.0,
        })
        ax = plt.gca()
        ax.set_facecolor("#f5f5f5")

        ego_trajectory = []
        for i in range(SIM_LOOP):
            planner_param = param
            # path = self.poly_trajectory(ego_x, ego_y, ego_speed, ob)
            # NOTE:path = self.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed, ob)
            path = self.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed,
                                        ob, ego_yaw=ego_yaw, ego_a=ego_a, ego_kappa=ego_kappa)

            ego_x = path.x[1]
            ego_y = path.y[1]
            ego_yaw = path.yaw[1]
            # ego_speed = path.speed[1]
            ego_speed += (path.speed[1] - ego_speed) * 0.2
            ego_a = path.a[1]
            ego_kappa = path.c[1]
            ego_trajectory.append([ego_x, ego_y, ego_yaw, ego_speed, ego_a, ego_kappa])

            if np.hypot(path.x[1] - self.wx[380], path.y[1] - self.wy[380]) <= 1.0:
                print("Goal")
                break

            if self.show_animation:
                plt.figure(3)
                plt.cla()
                plt.plot(self.road_left[:,0], self.road_left[:,1], color='#002060', label='lane Edge')
                plt.plot(self.road_right[:,0], self.road_right[:,1], color='#002060')
                plt.plot(self.road_center[:,0], self.road_center[:,1], '--', color='#002060')

                plt.plot(path.x[1:], path.y[1:], "-or", markersize=0.5)
                plt.plot(path.x[1], path.y[1], "vc")

                plt.grid(True)
                plt.pause(0.0001)
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        print("Finish")
        ego_trajectory = np.array(ego_trajectory)
        if self.show_animation:  # pragma: no cover
            plt.grid(True)
            plt.plot(ego_trajectory[:,0], ego_trajectory[:,1], "-r",marker='o', markersize=0.5, label='Ego Trajectory')
            plt.xlabel("X/m", fontsize=15)
            plt.ylabel("Y/m", fontsize=15)
            plt.title(f"k_J = {param[0]}, K_D = {param[1]}")
            plt.legend(loc='best', prop={'size': 12})
            plt.savefig(f"./figures/polyplanner/frenet_plan_global_kJ{param[0]}_kD{param[1]}.png", bbox_inches='tight')
            plt.show()
    
    def debug_sim_frenet_plan_frenet(self):
        ego_x = self.tx[0]
        ego_y = self.ty[0]
        ego_speed = 20.0 / 3.6  # current speed [m/s]
        ego_yaw = self.tyaw[0]
        ego_kappa = self.tc[0]
        ego_a = 0.0
        target_speed = 55.0 / 3.6
        ob = np.array([])
        # param = [0.0, 0.5]
        param = [0.2, 1.0]

        SIM_LOOP = 800 # simulation loop

        plt.figure(3)
        plt.rcParams['xtick.direction'] = 'in'  #将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'SimSun'],  # 英文字体为新罗马，中文字体为宋体
            'font.sans-serif': ['Times New Roman', 'SimSun'],  # 无衬线字体
            'font.serif': ['Times New Roman', 'SimSun'],  # 衬线字体
            'mathtext.fontset': 'custom', # 设置Latex字体为用户自定义, mathtext的字体是与font.sans-serif绑定的
            'mathtext.default': 'rm',  # 设置mathtext的默认字体为Times New Roman
            # 设置mathtext的无衬线字体为Times New Roman
            'mathtext.sf': 'Times New Roman',  # 设置mathtext的无衬线字体为Times New Roman
            'mathtext.rm': 'Times New Roman',  # 设置mathtext的衬线字体为Times New Roman
            'font.size': 12,  # 五号字
            'axes.unicode_minus': False,  # 解决负号显示问题
            'text.usetex': False,
            'figure.figsize': (3.5, 2.625), #单位是inches, 按IEEE RAL 要求，双栏图片(7.16, 5.37), 单栏图片(3.5, 2.625)
            'figure.dpi': 600,
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'lines.linewidth': 1.0,
        })
        ax = plt.gca()
        ax.set_facecolor("#f5f5f5")

        road_s = [[0]]
        offset_error = [[0]]

        for i in range(SIM_LOOP):
            planner_param = param
            # path = self.poly_trajectory(ego_x, ego_y, ego_speed, ob)
            # NOTE:path = self.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed, ob)
            start = time.time()
            path = self.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed,
                                        ob, ego_yaw=ego_yaw, ego_a=ego_a, ego_kappa=ego_kappa)
            end = time.time()
            print(f"Planning time: {(end - start)*1000:.2f} ms")     
            ego_x = path.x[1]
            ego_y = path.y[1]
            ego_yaw = path.yaw[1]
            ego_speed = path.speed[1]
            ego_a = path.a[1]
            ego_kappa = path.c[1]

            road_s.append([path.s[1]])
            offset_error.append([path.l[1]])

            if np.hypot(path.x[1] - self.wx[380], path.y[1] - self.wy[380]) <= 1.0:
                print("Goal")
                break

            # ## for more roadpoint 0.1m
            # if np.hypot(path.x[1] - self.wx[1895], path.y[1] - self.wy[1895]) <= 1.0:
            #     print("Goal")
            #     break

            if self.show_animation:
                plt.figure(3)
                plt.cla()
                plt.plot(np.linspace(0, 190, 100), np.ones(100)*round(road_width/2, 2), color='#002060', linewidth=1, label='lane Edge')
                plt.plot(np.linspace(0, 190, 100), np.zeros(100), '--',color='#002060', linewidth=1)
                plt.plot(np.linspace(0, 190, 100), np.ones(100)*-1*round(road_width/2, 2), color='#002060', linewidth=1)

                plt.plot(np.linspace(0, 190, 100), np.ones(100)*MAX_ROAD_WIDTH, '--', color='#006400',linewidth=1, label='Safety Zone Boundary')
                plt.plot(np.linspace(0, 190, 100), np.ones(100)*-1*MAX_ROAD_WIDTH, '--', color='#006400', linewidth=1)


                plt.plot(path.s[1:], path.l[1:], "-or", markersize=0.5)
                plt.plot(path.s[1], path.l[1], "vc", label=f"ego, speed = {path.speed[1]*3.6:.1f} km/h")
                plt.legend(loc='best', prop={'size': 12})
                # plt.title(f"speed = {path.speed[-1]*3.6:.1f} km/h", fontsize=12)

                plt.grid(True)
                plt.pause(0.0001)
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        print("Finish")
        if self.show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(np.array(road_s), np.array(offset_error), 'b')
            plt.ylim(-0.01, 0.01)
            plt.grid(True)
            plt.xlabel("S /m", fontsize=15)
            plt.ylabel("Lateral Error /m", fontsize=15)
            plt.title(f"k_J = {param[0]}, K_D = {param[1]}")
            plt.savefig("./figures/polyplanner/lateral_error.png")
            plt.show()

    def test_frenet_conversion_consistency(self):
        """
        闭环测试：Frenet -> Cartesian -> Frenet
        验证坐标转换函数的数学一致性
        """
        print("\n" + "="*20 + " 开始坐标转换闭环测试 " + "="*20)
    
        # 1. 构造一个测试用的初始 Frenet 状态
        # 假设我们在道路 100m 处，向左偏离 1.5m，纵向速度 10m/s，正在向中心线靠拢
        test_fp = FrenetPath()
        test_fp.s = [40.0 % self.ref_s[-1]]
        test_fp.l = [0.5]
        test_fp.s_dot = [10.0]
        test_fp.l_dot = [0.5]
        test_fp.s_ddot = [0.5]
        test_fp.l_ddot = [0.1]

        # 2. 【逆转换】Frenet -> Cartesian (手动计算作为输入)
        # 获取参考点属性
        global_fp = self.calc_global_paths(test_fp, self.csp)

        # 3. 【正转换】调用待测函数
        res_s, res_l, res_s_dot, res_l_dot, res_s_ddot, res_l_ddot = \
            self.cartesian_to_frenet_state(global_fp.x, global_fp.y, global_fp.yaw, global_fp.speed, global_fp.a, global_fp.c)

        # 4. 误差对比
        results = {
            "s": (test_fp.s[0], res_s),
            "l": (test_fp.l[0], res_l),
            "s_dot": (test_fp.s_dot[0], res_s_dot),
            "l_dot": (test_fp.l_dot[0], res_l_dot),
            "s_ddot": (test_fp.s_ddot[0], res_s_ddot),
            "l_ddot": (test_fp.l_ddot[0], res_l_ddot)
        }

        print(f"{'变量':<10} | {'预期值':<10} | {'实际值':<10} | {'误差':<10}")
        print("-" * 50)
        for var, (expected, actual) in results.items():
            error = abs(expected - actual)
            # s 因为环形可能存在 max_s 的倍数差异，做特殊处理
            if var == 's':
                error = min(error, abs(planner.ref_s[-1] - error))
            
            status = "✅ PASS" if error < 1e-4 else "❌ FAIL"
            print(f"{var:<10} | {expected:>10.4f} | {actual:>10.4f} | {error:>10.4e} {status}")

        print("="*50)

    def debug_sim_frenet_params_legend(self):
        # ==========================================
        # 1. 初始状态与参数设置
        # ==========================================
        ego_x, ego_y = self.tx[0], self.ty[0]
        ego_speed = 20.0 / 3.6  # [m/s]
        ego_yaw, ego_kappa = self.tyaw[0], self.tc[0]
        ego_a = 0.0
        target_speed = 55.0 / 3.6
        print("target_speed:", target_speed)
        print("min speed = ", target_speed - D_T_S * N_S_SAMPLE)
        ob = np.array([])
        
        # 构建参数列表 (当前演示遍历 K_D)
        # param_list = [[j_i, 1.0] for j_i in np.arange(0, 1.1, 0.1)] #check KJ
        # param_list = [[j_i, j_i] for j_i in np.arange(0, 1.1, 0.1)] # check KT
        param_list = [[0.0, d_i] for d_i in np.arange(0, 1.1, 0.1)] # check KD
        SIM_LOOP = len(param_list)

        # ==========================================
        # 2. 画布和全局样式配置
        # ==========================================
        plt.figure(1)
        plt.rcParams['xtick.direction'] = 'in'  #将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'SimSun'],  # 英文字体为新罗马，中文字体为宋体
            'font.sans-serif': ['Times New Roman', 'SimSun'],  # 无衬线字体
            'font.serif': ['Times New Roman', 'SimSun'],  # 衬线字体
            'mathtext.fontset': 'custom', # 设置Latex字体为用户自定义, mathtext的字体是与font.sans-serif绑定的
            'mathtext.default': 'rm',  # 设置mathtext的默认字体为Times New Roman
            # 设置mathtext的无衬线字体为Times New Roman
            'mathtext.sf': 'Times New Roman',  # 设置mathtext的无衬线字体为Times New Roman
            'mathtext.rm': 'Times New Roman',  # 设置mathtext的衬线字体为Times New Roman
            'font.size': 12,  # 五号字
            'axes.unicode_minus': False,  # 解决负号显示问题
            'text.usetex': False,
            'figure.figsize': (3.5, 2.625), #单位是inches, 按IEEE RAL 要求，双栏图片(7.16, 5.37), 单栏图片(3.5, 2.625)
            'figure.dpi': 600,
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'lines.linewidth': 1.0,
        })
        ax = plt.gca()
        ax.set_facecolor("#f5f5f5")

        plt.figure(1)
        plt.cla()
        plt.plot(np.linspace(0, 190, 100), np.ones(100)*round(road_width/2, 2), color='#002060', linewidth=1, label='lane Edge')
        plt.plot(np.linspace(0, 190, 100), np.zeros(100), '--',color='#002060', linewidth=1)
        plt.plot(np.linspace(0, 190, 100), np.ones(100)*-1*round(road_width/2, 2), color='#002060', linewidth=1)

        plt.plot(np.linspace(0, 190, 100), np.ones(100)*MAX_ROAD_WIDTH, '--', color='#006400',linewidth=1, label='Safety Zone Boundary')
        plt.plot(np.linspace(0, 190, 100), np.ones(100)*-1*MAX_ROAD_WIDTH, '--', color='#006400', linewidth=1)

        plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        for i, planner_param in enumerate(param_list):
            start = time.time()
            path = self.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed,
                                        ob, ego_yaw=ego_yaw, ego_a=ego_a, ego_kappa=ego_kappa)
            end = time.time()
            
            # 打印当前进度与耗时
            p_j, p_lat = planner_param[0], planner_param[1]
            print(f"[{i+1}/{SIM_LOOP}] Param(K_J={p_j:.1f}, P_lat={p_lat:.1f}) | Planning time: {(end - start)*1000:.2f} ms")     

            # 提取图例标签
            label = f"K_J = {p_j:.1f}, K_D = {p_lat:.1f}"
            # 绘制当前参数下的轨迹和速度线
            plt.plot(path.s, path.l, "-", label=label)
            plt.legend(loc='best', prop={'size': 12})

            # 实时更新画面 (如果只想看最终结果，可以将此行注释掉，生成速度会极快)
            plt.pause(0.0001)

        print("Finish!")
        plt.legend(loc='best', prop={'size': 12})
        plt.savefig("./figures/polyplanner/frenet_planner_kd.png", dpi=600, bbox_inches='tight')
        plt.show()
    def debug_sim_frenet_plan_params_speed(self):
        # ==========================================
        # 1. 初始状态与参数设置
        # ==========================================
        ego_x, ego_y = self.tx[0], self.ty[0]
        ego_speed = 20.0 / 3.6  # [m/s]
        ego_yaw, ego_kappa = self.tyaw[0], self.tc[0]
        ego_a = 0.0
        ob = np.array([])
        
        # 构建参数列表 (当前演示遍历 K_J)
        # param_list = [[j_i, 0.5] for j_i in np.arange(0, 1.1, 0.1)] # check KJ and KT
        param_list = [[0.0, 0.5], [1.0, 0.5]]
        target_speeds = [40.0 / 3.6, 45.0 / 3.6, 50.0 / 3.6, 55.0 / 3.6, 60.0 / 3.6]  # 不同目标速度下的表现

        # ==========================================
        # 2. 画布和全局样式配置
        # ==========================================
        plt.figure(1)
        plt.rcParams['xtick.direction'] = 'in'  #将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'SimSun'],  # 英文字体为新罗马，中文字体为宋体
            'font.sans-serif': ['Times New Roman', 'SimSun'],  # 无衬线字体
            'font.serif': ['Times New Roman', 'SimSun'],  # 衬线字体
            'mathtext.fontset': 'custom', # 设置Latex字体为用户自定义, mathtext的字体是与font.sans-serif绑定的
            'mathtext.default': 'rm',  # 设置mathtext的默认字体为Times New Roman
            # 设置mathtext的无衬线字体为Times New Roman
            'mathtext.sf': 'Times New Roman',  # 设置mathtext的无衬线字体为Times New Roman
            'mathtext.rm': 'Times New Roman',  # 设置mathtext的衬线字体为Times New Roman
            'font.size': 12,  # 五号字
            'axes.unicode_minus': False,  # 解决负号显示问题
            'text.usetex': False,
            'figure.figsize': (3.5, 2.625), #单位是inches, 按IEEE RAL 要求，双栏图片(7.16, 5.37), 单栏图片(3.5, 2.625)
            'figure.dpi': 600,
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'lines.linewidth': 1.0,
        })
        ax = plt.gca()
        ax.set_facecolor("#f5f5f5")

        plt.figure(1)
        plt.cla()

        plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        for planner_param in param_list:
            for target_speed in target_speeds:
                print("target_speed:", target_speed)
                print("min speed = ", target_speed - D_T_S * N_S_SAMPLE)
                path = self.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed,
                                            ob, ego_yaw=ego_yaw, ego_a=ego_a, ego_kappa=ego_kappa)

                # 提取图例标签
                label = f"target speed = {target_speed:.1f}"
                plt.plot(path.s, path.speed, "-",label=label)
                # 实时更新画面 (如果只想看最终结果，可以将此行注释掉，生成速度会极快)
                plt.pause(0.0001)

        print("Finish!")
        plt.legend(loc='best', prop={'size': 12})
        plt.xlabel("S /m", fontsize=15)
        plt.ylabel("Speed m/s", fontsize=15)
        plt.savefig("./figures/polyplanner/frenet_planner_speed.png", dpi=600, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    env_data = natural_road_load()
    planner = Polyplanner(env_data, lane_id=1)
    # planner.test_frenet_conversion_consistency()
    # planner.debug_sim_frenet_plan_global()
    # planner.debug_sim_frenet_plan_frenet()
    # planner.debug_sim_frenet_params_legend()
    # planner.debug_sim_frenet_plan_params_speed()

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
    plt.rcParams['font.size'] = 13             # 全局基准字号
    plt.rcParams['xtick.direction'] = 'in'     # X轴刻度线向内
    plt.rcParams['ytick.direction'] = 'in'     # Y轴刻度线向内
    # ==============================================================================
    # 研究路段曲率图
    # ==============================================================================
    # --- 1. 定义顶刊配色 ---
    color_straight = '#34495E'  # 直线：深邃沥青蓝
    color_clothoid = '#F39C12'  # 回旋线：学术暗金橙
    color_curve    = '#8E44AD'  # 定曲率段：典雅紫藤色

    plt.figure(2603302)
    plt.plot(planner.ts[:300], planner.tc[:300], color=color_straight, label='直线段曲率')
    plt.plot(planner.ts[300:700], planner.tc[300:700], color=color_clothoid, label='回旋线段曲率')
    plt.plot(planner.ts[700:1200], planner.tc[700:1200], color=color_curve, label='定曲率段曲率')
    plt.plot(planner.ts[1200:1600], planner.tc[1200:1600], color=color_clothoid)
    plt.plot(planner.ts[1600:1870], planner.tc[1600:1870], color=color_straight)
    plt.plot(planner.ts[1870:1900], np.array(planner.tc[1870:1900])* 0.0, color=color_straight)
    plt.xlim(0, 200)
    plt.ylim(-0.01, 0.002)
    plt.xlabel('纵向距离 $s\mathrm{/m}$', fontsize=15)
    plt.ylabel('曲率 $\mathrm{\kappa}\mathrm{/(m^{-1})}$', fontsize=15)
    plt.legend(loc='best', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()  # 放在最后，自动调整所有间距
    # plt.savefig('./Figures/bend_road_curvature.png', dpi=600)  # Save the figure with high resolution
    plt.show()


