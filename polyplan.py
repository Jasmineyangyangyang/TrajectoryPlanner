import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import math
from CubicSpline import cubic_spline_planner
from global_road import natural_road_load

"""
单车道宽度：3.75m
车宽：1.85m
车辆允许的单侧max offset = 3.75/2 - 1.85/2 = 0.95m
"""

# Parameter
MAX_SPEED = 40.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
#----#
road_width = 3.75 #m
vehicle_width = 1.85 #m
offset_buffer = 0.15 #m, buffer for offset
MAX_ROAD_WIDTH = round(road_width/2 - vehicle_width/2 - offset_buffer, 2)  # maximum road width [m]  int(4/2-1.85/2) 车道减去车宽
D_ROAD_W = 0.2  # road width sampling length [m]
DT = 0.3  # firts searching time tick [s]
DT_best_path = 0.02  # best path searching time tick [s]
PLAN_T = 5.0  # max prediction time [m]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 2  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]块

# init cost weights
K_J = 0.0
K_D = 0.0
K_T = 1.0 - K_J - K_D
K_LAT = 1.0
K_LON = K_LAT

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
        self.d = []     # lateral position
        self.d_d = []   # lateral speed
        self.d_dd = []  # lateral acceleration
        self.d_ddd = [] # lateral jerk
        self.s = []     # longitudinal position
        self.s_d = []   # longitudinal speed
        self.s_dd = []  # longitudinal acceleration
        self.s_ddd = [] # longitudinal jerk
        self.cd = 0.0   # cost d
        self.cv = 0.0   # cost v
        self.cf = 0.0   # cost sum

        self.x = []     # global x position
        self.y = []     # global y position
        self.yaw = []   # global yaw position
        self.speed = []    # global delta s
        self.c = []     # global curvature

        self.lat_param = []  # lateral motion planning parameter
        self.lon_param = []  # longitudinal motion planning parameter

def calc_frenet_path(lat_param, lon_param):
    fp = FrenetPath()
    lat_qp = QuinticPolynomial(lat_param[0], lat_param[1], lat_param[2], lat_param[3], lat_param[4])
    lon_qp = QuarticPolynomial(lon_param[0], lon_param[1], lon_param[2], lon_param[3], lon_param[4])
    
    fp.t = [t for t in np.arange(0.0, PLAN_T, DT_best_path)]

    fp.d = [lat_qp.calc_point(t) for t in fp.t]
    fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
    fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
    fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

    fp.s = [lon_qp.calc_point(t) for t in fp.t]
    fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
    fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
    fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

    return fp
    
def calc_frenet_paths(csp, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, planner_param, TARGET_SPEED):
    """# target speed [m/s]"""
    MAX_JERK_2= 0.5  # maximum jerk error[m/sss]
    MAX_SPEED_2 = (TARGET_SPEED-40/3.6)**2  # maximum speed error[m/s]
    MAX_FRENET_2 = (TARGET_SPEED*5.-40/3.6*5.0)**2 # maximum frenet error[m]

    frenet_paths = []
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH+D_ROAD_W, D_ROAD_W):
    # Lateral motion planning
        Ti = PLAN_T   # acctually you can search different Ti, but that will cost more time
        fp = FrenetPath()
        LEN_PATH = np.arange(0.0, Ti+DT, DT).shape[0]

        lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di,  Ti)  # 低速时将Ti理解成Si

        fp.t = [t for t in np.arange(0.0, Ti+DT, DT)]
        fp.d = [lat_qp.calc_point(t) for t in fp.t]
        fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
        fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
        fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]
        fp.lat_param = [c_d, c_d_d, c_d_dd, di, Ti]

        # Longitudinal motion planning (Velocity keeping)
        for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                            TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
            tfp = copy.deepcopy(fp)

            lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, Ti)

            tfp.s = [lon_qp.calc_point(t) for t in fp.t]
            tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            tfp.lon_param = [s0, c_speed, c_accel, tv, Ti]

            Jp = sum(np.power(tfp.d_ddd, 2))  # square of lat jerk
            Js = sum(np.power(tfp.s_ddd, 2))  # square of lon jerk

            # square of diff from target speed
            ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

            # square of diff from target frenet distance
            df = (tfp.s[-1]-TARGET_SPEED*Ti)**2

            # tfp.cd = planner_param[0] * Jp + planner_param[1] * tfp.d[-1] ** 2  + (1.0 - planner_param[0] - planner_param[1]) * Ti
            # tfp.cv = planner_param[0] * Js + planner_param[1] * ds + (1.0 - planner_param[0] - planner_param[1]) * Ti

            # Normalize data to within [-1,1] : 2(x-min)/(max-min)-1
            Jp = 2 * Jp / (MAX_JERK_2*LEN_PATH) -1
            Js = 2 * Js / (MAX_JERK_2*LEN_PATH) -1
            ds = 2 * ds / MAX_SPEED_2 -1 if MAX_SPEED_2 > 1e-3 else ds
            df = 2 * df / MAX_FRENET_2 -1 if MAX_FRENET_2 > 1e-3 else df

            cdd = 2 * (tfp.d[-1] ** 2) / (MAX_ROAD_WIDTH ** 2) - 1
            coffset = 2 * (tfp.d[-1] - (-MAX_ROAD_WIDTH)) / (MAX_ROAD_WIDTH - (-MAX_ROAD_WIDTH)) - 1
            tfp.cd = planner_param[0] * Jp + planner_param[1] * cdd + (1.0 - planner_param[0] - planner_param[1]) * coffset
            tfp.cv = planner_param[0] * Js + planner_param[1] * ds + (1.0 - planner_param[0] - planner_param[1]) * df
            tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

            frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fp, csp):
    # calc global positions
    for i in range(len(fp.s)):
        if fp.s[i] <= 0:
            ix, iy = csp.calc_position(0)
            i_yaw = csp.calc_yaw(0)
        else:
            ix, iy = csp.calc_position(fp.s[i])
            i_yaw = csp.calc_yaw(fp.s[i])
        if ix is None:
            break
        
        di = fp.d[i]
        fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
        fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
        fp.x.append(fx)
        fp.y.append(fy)

    # calc yaw and ds
    for i in range(len(fp.x) - 1):
        dx = fp.x[i + 1] - fp.x[i]
        dy = fp.y[i + 1] - fp.y[i]
        fp.yaw.append(math.atan2(dy, dx))
        fp.speed.append(math.hypot(dx, dy))
    if len(fp.yaw) != 0:
        fp.yaw.append(fp.yaw[-1])
        fp.speed.append(fp.speed[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.speed[i])
        fp.c.append(fp.c[-1])

    return fp


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
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    # return [fplist[i] for i in ok_ind]
    return ok_ind


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, planner_param, target_speed, ob):
    fplist = calc_frenet_paths(csp, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, planner_param, target_speed)
    # fplist_ok_ind = check_paths(fplist, ob)  # check maximum speed, accel, curvature, collision
    # fplist = [fplist[i] for i in fplist_ok_ind]
    # for i in range(len(fplist)):
    #     plt.plot(fplist[i].s, fplist[i].d, label=f"{fplist[i].cf:.2f}")
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
    fp = calc_global_paths(frenet_fp, csp)

    return fp

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

    return rx, ry, ryaw, rk, csp


class Polyplanner():
    def __init__(self, env_data, lane_id):

        self.env_data = env_data
        # _, self.road, _ = self.env_data.build_trajectory()
        self.road = self.env_data.read_from_csv()
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
        self.tx, self.ty, self.tyaw, self.tc, self.csp = generate_target_course(self.wx, self.wy)
        self.show_animation = True
    
    def find_nearest_point(self, x, y):
        # 计算每个参考点到 (x, y) 的距离
        distances = np.sqrt((np.array(self.tx) - x)**2 + (np.array(self.ty) - y)**2)
        # 找到最近的参考点的索引
        nearest_index = np.argmin(distances)
        
        return nearest_index
    def calculate_frenet_coordinates(self, x, y, ego_yaw, speed, ego_kappa, ego_a):
        # 找到最近的参考点
        nearest_index = self.find_nearest_point(x, y)
        
        # 获取最近点的坐标
        cx = self.tx[nearest_index]
        cy = self.ty[nearest_index]
        
        # 获取最近点的累积距离 s
        s = 0.1 *nearest_index
        
        # 计算投影点到 (x, y) 的垂直距离 d
        dx = x - cx
        dy = y - cy
        ds = dx * np.cos(self.tyaw[nearest_index]) + dy * np.sin(self.tyaw[nearest_index])
        s = s + ds
        d = -dx * np.sin(self.tyaw[nearest_index]) + dy * np.cos(self.tyaw[nearest_index])
        
        c_speed = speed
        c_accel = 0.0
        c_d_d = 0.0
        c_d_dd = 0.0
        return s, c_speed, c_accel, d, c_d_d, c_d_dd

    def poly_trajectory(self, ego_x, ego_y, ego_speed, planner_param, target_speed, ob, ego_yaw, ego_a, ego_kappa):
        s0, c_speed, c_accel, c_d, c_d_d, c_d_dd = self.calculate_frenet_coordinates( ego_x, ego_y, ego_yaw, ego_speed, ego_kappa, ego_a)
        # pathlist, best_id = frenet_optimal_planning(self.csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, planner_param, ob)
        # return pathlist, best_id
        # print(f"path target = {target_speed}")
        path = frenet_optimal_planning(self.csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, planner_param, target_speed, ob)
        # print(f"plan result {path.s_d[1]}")
        return path
    
    def debug_sim_global_all(self):
        ego_x = self.tx[0]
        ego_y = self.ty[0]
        ego_speed = 42.0 / 3.6  # current speed [m/s]
        planner_param = [K_J, K_D] # K_J, K_T, K_D, K_LAT, K_LON
        target_speed = 50.0 / 3.6
        ob = np.array([])

        param_list = []
        # param_list.append(planner_param)
        param_list.append([0.5, 0.5])

        SIM_LOOP = 181 # simulation loop

        if self.show_animation:  # pragma: no cover
            plt.figure(12)
            plt.rcParams['xtick.direction'] = 'in'  #将x轴的刻度线方向设置向内
            plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内
            ax = plt.gca()
            ax.set_facecolor("grey")
            plt.plot(self.road[:,0], self.road[:,1], 'b')
            plt.plot(self.road[:,2], self.road[:,3], 'b')
            plt.plot(self.road[:,4], self.road[:,5], 'b')
            plt.plot(self.road_center[:380,0], self.road_center[:380,1], 'b--.')

        for param in param_list:
            planner_param = param
            print(planner_param)
            x = []
            y = []
            velocity = []

            for i in range(SIM_LOOP):
                # start_time = time.time()
                path = self.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed, ob)  # add ego 0.1s后的state
                # end_time = time.time()
                # print(f"Time of bend cost: {end_time - start_time} seconds")    
                # if(len(path.x) != 50):
                #     print(len(path.x))
                if path is not None:
                    c_speed = path.s_d[1]
                    x.append(path.x[1])
                    y.append(path.y[1])
                    velocity.append(c_speed)
        
                    ego_x = path.x[1]
                    ego_y = path.y[1]
                    ego_speed = path.s_d[1]

                    if np.hypot(path.x[1] - self.tx[1880], path.y[1] - self.ty[1880]) <= 1.0:  # 1880 is the 380 point of the reference trajectory
                        print("Goal")
                        break
                else:
                    print(f"loop {i} Path is None")

            if self.show_animation:
                plt.figure(11)
                plt.plot(velocity)
                plt.figure(12)
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(self.tx, self.ty)
                if not len(ob) == 0:
                    plt.plot(ob[:, 0], ob[:, 1], "xk")
                labelstr = 'K_J=' + str(planner_param[0]) + ', K_D=' + str(planner_param[1]) + ', K_T=' + str(1.0-planner_param[0]-planner_param[1]) 
                plt.plot(x, y, 'r.', label=labelstr)
                plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
                plt.grid(True)
                plt.pause(0.0001)
                    
        print("Finish")
        if self.show_animation:  # pragma: no cover
            plt.grid(True)
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend()
            plt.pause(0.0001)
            plt.show()
            plt.savefig('global.png')

    def debug_sim_frenet_all(self):
        ego_x = self.tx[0]
        ego_y = self.ty[0]
        ego_speed = 42.0 / 3.6  # current speed [m/s]
        planner_param = [K_J, K_D] # K_J, K_T, K_D, K_LAT, K_LON
        target_speed = 50.0 / 3.6
        ob = np.array([])

        param_list = []
        # param_list.append(planner_param)
        param_list.append([0.5, 0.5])

        SIM_LOOP = 95 # simulation loop

        plt.figure(12)
        plt.rcParams['xtick.direction'] = 'in'  #将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内
        ax = plt.gca()
        ax.set_facecolor("grey")

        plt.plot(np.linspace(0, 430, 100), np.ones(100)*2.0, 'k', linewidth=1)
        plt.plot(np.linspace(0, 430, 100), np.zeros(100), 'k--', linewidth=1)
        plt.plot(np.linspace(0, 430, 100), np.ones(100)*-2.0, 'k', linewidth=1)

        plt.plot(np.linspace(0, 430, 100), np.ones(100)*0.83, 'k--', linewidth=1)
        plt.plot(np.linspace(0, 430, 100), np.ones(100)*-0.83, 'k--', linewidth=1)
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        
        for param in param_list:
            planner_param = param
            print(planner_param)
            x = []
            y = []
            velocity = []

            for i in range(SIM_LOOP):
                # path = self.poly_trajectory(ego_x, ego_y, ego_speed, ob)
                pathlist = self.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed, ob)
                c_speed = pathlist.s_d[1]
                x.append(pathlist.s[1])
                y.append(pathlist.d[1])
                velocity.append(c_speed)
    
                ego_x = pathlist.x[1]
                ego_y = pathlist.y[1]
                ego_speed = pathlist.s_d[1]

                if np.hypot(pathlist.x[1] - self.tx[-1], pathlist.y[1] - self.ty[-1]) <= 1.0:
                    print("Goal")
                    break
            
                # print('mean velocity=', np.mean(np.array(velocity)))
                # print('mean offset=', np.mean(np.array(y)))
                if self.show_animation:
                    # for stopping simulation with the esc key.
                    plt.gcf().canvas.mpl_connect(
                        'key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                    
                    plt.figure(13)
                    # labelstr = 'K_J=' + str(planner_param[0]) + ', K_D=' + str(planner_param[1]) + ', K_T=' + str(1.0-planner_param[0]-planner_param[1]) 
                    # plt.plot(velocity, label=labelstr)
                    # plt.legend()
                    plt.plot(velocity)
                    plt.gcf().canvas.mpl_connect(
                        'key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                    plt.pause(0.0001)

                    plt.figure(12)
                    # plt.plot(x, y, label=labelstr)
                    plt.plot(pathlist.s, pathlist.d)
                    plt.plot(pathlist.s[1], pathlist.d[1], 'vc')
                    plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
                    plt.grid(True)
                    # plt.legend()
                    plt.pause(0.0001)

        print("Finish")
        if self.show_animation:  # pragma: no cover
            plt.figure(12)
            plt.savefig('frenet_all_trajectory.png')
            plt.figure(13)
            plt.savefig('frenet_all_velocity.png')
            plt.show()

    def debug_sim_frenet_planT(self):
        ego_x = self.tx[0]
        ego_y = self.ty[0]
        ego_speed = 40.0 / 3.6  # current speed [m/s]
        planner_param = [K_J, K_D] # K_J, K_T, K_D, K_LAT, K_LON
        target_speed = 55.0 / 3.6
        ob = np.array([])
        param_list = []

        # K_J范围测试
        # #---------------- #
        # param_list.append([0.7, 0.0])   # mean velocity= 11.1 m/s, end offset= -0.75 m
        # param_list.append([0.8, 0.0])    # mean velocity= 11.1 m/s, end offset= -0.50 m
        # param_list.append([0.9, 0.0])    # mean velocity= 11.1 m/s, end offset= -0.25 m
        # param_list.append([1.0, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.0 m
        # param_list.append([1.1, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.25 m
        # param_list.append([1.2, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.25 m
        # param_list.append([1.3, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.50 m
        # param_list.append([1.4, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.50 m
        # param_list.append([1.5, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.75 m
        # param_list.append([1.6, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.75 m

        # K_D范围测试
        # param_list.append([0.0, 0.1])   # mean velocity= 13.1 m/s, end offset= -1.0 m
        # param_list.append([0.0, 0.2])   # mean velocity= 13.1 m/s, end offset= -1.0 m
        # param_list.append([0.0, 0.3])   # mean velocity= 12.4 m/s, end offset= -1.0 m
        # param_list.append([0.0, 0.4])   # mean velocity= 12.4 m/s, end offset= -1.0 m
        # param_list.append([0.0, 0.5])   # mean velocity= 12.4 m/s, end offset= -1.0 m
        # param_list.append([0.0, 0.6])   # mean velocity= 12.4 m/s, end offset= -1.0 m
        # param_list.append([0.0, 0.7])   # mean velocity= 12.4 m/s, end offset= -0.25 m

        param_list.append([0.4, 0.1])    # mean velocity= 11.1 m/s, end offset= 0.25 m
        param_list.append([0.0, 1.0])    # mean velocity= 11.1 m/s, end offset= 0.25 m
        # param_list.append([1.3, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.50 m
        # param_list.append([1.4, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.50 m
        # param_list.append([1.5, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.75 m
        # param_list.append([1.6, 0.0])    # mean velocity= 11.1 m/s, end offset= 0.75 m

        # #for K_d  
        # k_d = 0.0     # mean velocity= 11.3 m/s
        # param_list.append([0.75, k_d])   # end offset= -0.75 m
        # param_list.append([0.8, k_d])   #  end offset= -0.45 m
        # param_list.append([0.9, k_d])   #  end offset= -0.25 m
        # param_list.append([1.0, k_d])   #  end offset=  0.0 m
        # param_list.append([1.1, k_d])   #  end offset=  0.25 m
        # param_list.append([1.2, k_d])   #  end offset=  0.45 m
        # param_list.append([1.4, k_d])   #  end offset=  0.75 m

        #for K_J
        # k_j= 0.1  # mean velocity= 12.7 m/s
        # # k_j= -0.1  # mean velocity= 13.4 m/s
        # param_list.append([k_j, 0.1])   #  end offset= -0.8 m
        # param_list.append([k_j, 0.2])   #  end offset= -0.75 m
        # param_list.append([k_j, 0.3])   #  end offset= -0.45 m
        # param_list.append([k_j, 0.4])   #  end offset= -0.25 m
        # param_list.append([k_j, 0.7])   #  end offset=  0.0 m
        # param_list.append([k_j, 3.0])   #  end offset=  0.25 m

        # k_j= 0.5  # mean velocity= 12.0 m/s
        # param_list.append([k_j, 0.3])   #  end offset= -0.45 m
        # param_list.append([k_j, 2.0])     #  end offset= 0.8 m

        # k_j= 1.5  # mean velocity= 12.7 m/s
        # param_list.append([k_j, 1.0])   #  end offset=  0.0 m
        # param_list.append([k_j, 2.0])   #  end offset=  0.0 m
        # param_list.append([k_j, 3.0])   #  end offset=  0.0 m
        
        # param_list.append([k_j, 5.0])   #  end offset=  0.0 m
        # param_list.append([k_j, 6.0])   #  end offset=  0.0 m

        # # # for online update
        # param_list.append([0.0, 1.0])   # mean velocity= 12.7 m/s, end offset= 0.0 m [K_T=0.0]
        # param_list.append([2.0, 0.0])   # mean velocity= 12.7 m/s, end offset= 0.0 m [K_T=0.0]
        # param_list.append([0.75, 0.0])   # mean velocity= 12.7 m/s, end offset= 0.0 m [K_T=0.0]
        # param_list.append([0.0274, 0.9863])   # mean velocity= 12.7 m/s, end offset= 0.0 m [K_T=0.0]
        # param_list.append([0.0145, 0.9927])   # mean velocity= 12.7 m/s, end offset= 0.0 m [K_T=0.0]
        # param_list.append([0.8906, 0.0])   # mean velocity= 12.7 m/s, end offset= 0.0 m [K_T=0.0]

        # for parameters plan show
        # k_t = 0.1
        # param_list.append([1-k_t-0.1, 0.1])   # mean velocity= 13.1 m/s, end offset= -1.0 m
        # param_list.append([1-k_t-0.2, 0.2])   # mean velocity= 13.1 m/s, end offset= -1.0 m
        # param_list.append([1-k_t-0.3, 0.3])   # mean velocity= 12.4 m/s, end offset= -1.0 m
        # param_list.append([1-k_t-0.7, 0.7])   # mean velocity= 12.4 m/s, end offset= -0.25 m

        SIM_LOOP = len(param_list) # simulation loop

        plt.figure(12)
        plt.rcParams['xtick.direction'] = 'in'  #将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内
        ax = plt.gca()
        ax.set_facecolor("grey")

        plt.plot(np.linspace(0, 191, 100), np.ones(100)*2.0, 'k', linewidth=1)
        plt.plot(np.linspace(0, 191, 100), np.zeros(100), 'k--', linewidth=1)
        plt.plot(np.linspace(0, 191, 100), np.ones(100)*-2.0, 'k', linewidth=1)

        plt.plot(np.linspace(0, 191, 100), np.ones(100)*0.83, '--', color='lime',linewidth=1)
        plt.plot(np.linspace(0, 191, 100), np.ones(100)*-0.83, '--', color='lime', linewidth=1)

        for i in range(SIM_LOOP):
            planner_param = param_list[i]
            # path = self.poly_trajectory(ego_x, ego_y, ego_speed, ob)
            ego_yaw=0.0
            ego_a=0.0
            ego_kappa=0.0
            path = self.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed, 
                                                    ob, ego_yaw, ego_a, ego_kappa)
            end_offset = path.d[-1]
            x = path.s
            y = path.d
            velocity = path.s_d

            # ego_x = path.x[1]
            # ego_y = path.y[1]
            # ego_speed = path.s_d[1]

            if np.hypot(path.x[1] - self.wx[380], path.y[1] - self.wy[380]) <= 1.0:
                print("Goal")
                break

            if self.show_animation:
                plt.figure(13)
                plt.plot(x, velocity, label=f"K_T={1.0-planner_param[0]-planner_param[1]:.2f}, v=" + str(np.mean(velocity))[0:4])
                plt.title("speed[m/s]:")
                plt.ylim([5, 16])
                plt.legend()
                plt.pause(0.0001)
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                plt.figure(12)
                labelstr = 'K_J=' + str(planner_param[0]) + ', K_D=' + str(planner_param[1]) + ', K_T=' + f"{1.0-planner_param[0]-planner_param[1]:.2f}"
                plt.plot(x, y, label=labelstr)
                plt.plot(x[1], y[1], 'vc')
                plt.title("end offset[m]:" + f"{end_offset:.4f}")  #round(end_offset, 4)
                plt.grid(True)
                plt.pause(0.0001)
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        print("Finish")
        if self.show_animation:  # pragma: no cover
            plt.grid(True)
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend()
            plt.pause(0.0001)
            # plt.savefig('frenet.png')
            plt.show()


if __name__ == '__main__':
    env_data = natural_road_load()
    planner = Polyplanner(env_data, lane_id=1)
    # planner.debug_sim_global_all()
    # planner.debug_sim_frenet_all()
    planner.debug_sim_frenet_planT()
