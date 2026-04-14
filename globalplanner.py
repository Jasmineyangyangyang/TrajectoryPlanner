# from traject_plan_control.polyplan import Polyplanner
# from traject_plan_control.roadplan import Roadplanner
# from traject_plan_control.global_road import natural_road_load
# import matplotlib.pyplot as plt

# for debug
# from polyplan_States import Polyplanner  # old [-1, 1]cost, full states coordinate transformation
from polyplan_States_cost import Polyplanner  # new [0, 1]cost
from roadplan import Roadplanner
import matplotlib.pyplot as plt
from global_road import natural_road_load
import time
import numpy as np
import copy
        
class GlobalPlanner():
    def __init__(self, road_env, lane_id):
        self.env_data = road_env    # road environment
        self.road = self.env_data.read_from_csv('./')
        self.ego_lane_id = lane_id
        # self.planner_param_init = [0.0, 0.5] # center line
        # self.planner_param_init = [0.0, 1.0] # inner
        self.planner_param_init = [0.2, 1.0] # inner

        self.road_tmp = []
        print("=============Set polyplanner=============")
        self.polyplanner = Polyplanner(self.env_data, self.ego_lane_id)
        print("=============Set roadplanner=============")
        self.roadplanner = Roadplanner(self.env_data, self.ego_lane_id)
        self.show_animation = True
    
    # def generate_trajectory(self, ego_x, ego_y, ego_speed, polyplan_trigger, planner_param,next_x, next_y,next_speed):
    def generate_trajectory(self, ego_x, ego_y, ego_speed, ego_yaw, ego_a, ego_kappa, polyplan_trigger, planner_param, target_speed=55.0 / 3.6, ob = []):
        if polyplan_trigger:
            # start_time = time.time()
            path = self.polyplanner.poly_trajectory(ego_x, ego_y, ego_speed, planner_param, target_speed, 
                                                    ob, ego_yaw, ego_a, ego_kappa)
            # end_time = time.time()
            # print("Polyplanner time: ", end_time - start_time)
            # path_list = [path.x, path.y]
            # with open('path_list.txt', 'a') as f:
            #     f.write(str(path_list))
            #     f.write('\n')
        else:
            path = self.roadplanner.road_trajectory(ego_x, ego_y, ego_speed)
        
        # print("path.x: ", len(path.x))
        return path
    
    def calc_curvature(self, x1, y1, x2, y2, x3, y3):
        """
        三点圆曲率公式， kappa = 2 * A/abc, A是三角形面积
        """
        a = np.hypot(x1 - x2, y1 - y2)
        b = np.hypot(x2 - x3, y2 - y3)
        c = np.hypot(x3 - x1, y3 - y1)

        area = abs(
            x1*(y2-y3) +
            x2*(y3-y1) +
            x3*(y1-y2)
        ) / 2.0

        if area < 1e-6:
            return 0.0

        # 使用向量叉乘判断转向方向
        # 向量1: P1 -> P2, 向量2: P2 -> P3
        cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        sign = 1.0 if cross_product >= 0 else -1.0

        # 乘以方向符号
        curvature = sign * (4 * area / (a * b * c))

        return curvature

    def stanley_controller(self, ego_x, ego_y, ego_yaw, ego_speed, path, wheelBase=2.6):
        """
        规划器给的path是质心位置的ego pose，
        运动学模型为后轴中心，所以需要计算前轴位置；
        台架试验是动力学模型，ego pose为质心，front_x = ego_x + wheelBase / 2 * np.cos(ego_yaw)
        """
        front_x = ego_x + wheelBase * np.cos(ego_yaw)  
        front_y = ego_y + wheelBase * np.sin(ego_yaw)
        currentIndex = 0
        road_len = len(path.x)
        # numofpoints = 1  # 用于Stanley控制器的前后预瞄点数量,用来计算平均道路曲率和航向角，不平均信号会抖动
        lookahead_dist = 2.0  # 前瞄距离，用于计算前瞄点的索引，进而计算曲率
        # stanley_diffyaw_k = 1.5
        # stanley_ey_k = 10.0
        # stanley_curvature_k = 1.0
        # stanley_diffy_k = 2.65

        stanley_diffyaw_k = 1.0
        stanley_ey_k = 5.5
        stanley_curvature_k = 0.0  #0.31 右转为负，左转为正
        # stanley_curvature_k = 1.0  #0.31 右转为负，左转为正
        stanley_diffy_k = 1.65
        

        # stanley control constant
        comfort_acc = 2.0  # [m/s^2]
        dl_time = 1.5  # [s]
        abs_max_steer = np.pi / 4.0  # [rad]
        # find nearest point index
        currentIndex = self.find_nearest_point(front_x, front_y, path)
        # currentIndex = self.find_lookahead_point(front_x, front_y, path, 5.0)
        # frontIndex = min(currentIndex + numofpoints, road_len - 1)
        # rearIndex = max(currentIndex - numofpoints, 0)
        # psi_front = self.pi_2_pi(np.arctan2(path.y[frontIndex] - path.y[currentIndex], path.x[frontIndex] - path.x[currentIndex]))
        # psi_rear = self.pi_2_pi(np.arctan2(path.y[currentIndex] - path.y[rearIndex], path.x[currentIndex] - path.x[rearIndex]))
        # distFront = np.sqrt((path.x[currentIndex] - path.x[frontIndex])**2 + (path.y[currentIndex] - path.y[frontIndex])**2)
        # distRear = np.sqrt((path.x[currentIndex] - path.x[rearIndex])**2 + (path.y[currentIndex]- path.y[rearIndex])**2)

        # distRearFront = np.sqrt((path.x[frontIndex] - path.x[rearIndex])**2 + (path.y[frontIndex] - path.y[rearIndex])**2)
        # rearFront = [path.x[frontIndex]-path.x[rearIndex], path.y[frontIndex]-path.y[rearIndex]]
        # egoFront = [path.x[frontIndex] - front_x, path.y[frontIndex] - front_y]
        # coefFront = 0.25*(egoFront[0]*rearFront[0] + egoFront[1]*rearFront[1]) / (distRearFront**2)
        # roadpsi = coefFront * psi_rear + (1 - coefFront) * psi_front
        roadpsi = path.yaw[currentIndex]
        roadpsi = self.pi_2_pi(roadpsi)

        if currentIndex + 1 >= len(path.x):
            v1 = np.array([path.x[currentIndex], path.y[currentIndex]]) - np.array([path.x[currentIndex-1], path.y[currentIndex-1]])
        else:
            v1 = np.array([path.x[currentIndex+1], path.y[currentIndex+1]]) - np.array([path.x[currentIndex], path.y[currentIndex]])
        v2 = np.array([front_x, front_y]) - np.array([path.x[currentIndex], path.y[currentIndex]])
        tmpV = np.cross(v1, v2)
        Distance =  tmpV / max(np.linalg.norm(v1), 0.001)

        # if len(path.c)> 0:
        #     curvature = path.c[frontIndex]
        # else:
        #     curvature = (psi_front - psi_rear) / (distFront + distRear)

        if len(path.c) > 0:
            curvature = path.c[currentIndex]
        else:
            # 全局路段：往前后各找 2 米的索引
            frontIndex = self.find_lookahead_point(path.x[currentIndex], path.y[currentIndex], path, lookahead_dist)
            # 因为你没有写向后找的函数，可以直接用对称的索引差，或者更严谨点也写个 find_lookbehind_point
            index_diff = frontIndex - currentIndex
            rearIndex = max(currentIndex - index_diff, 0)
            
            curvature = self.calc_curvature(
                path.x[rearIndex], path.y[rearIndex],
                path.x[currentIndex], path.y[currentIndex],
                path.x[frontIndex], path.y[frontIndex]
            )

        psi_diff = self.calrealpsi(roadpsi, ego_yaw)
        # psi_diff = self.calrealpsi(ego_yaw, roadpsi)
        v_soft = 0.01
        ey_diff = self.pi_2_pi(np.arctan2(-1 * stanley_diffy_k*Distance, ego_speed*dl_time + v_soft))
        # curvature_diff = self.pi_2_pi(np.arctan(wheelBase*curvature))
        print("curvture = ", curvature)
        curvature_diff = self.pi_2_pi(np.arctan(np.clip(wheelBase * curvature, -0.5, 0.5)))
        # stanley steer
        print("psi_diff = ", psi_diff)
        print("ey_diff = ", ey_diff)
        print("curvature_diff = ", curvature_diff)
        # stanley_steer = stanley_diffyaw_k*psi_diff
        # stanley_steer = stanley_ey_k * ey_diff
        # stanley_steer = stanley_curvature_k* curvature_diff
        # stanley_steer = stanley_diffyaw_k*psi_diff + stanley_ey_k * ey_diff
        stanley_steer = stanley_diffyaw_k*psi_diff + stanley_ey_k * ey_diff + stanley_curvature_k* curvature_diff
        print("control_steer 1 psi_diff = ", stanley_diffyaw_k*psi_diff)
        print("control_steer 2 ey = ",  stanley_ey_k * ey_diff)
        print("control_steer 3 curvature = ", stanley_curvature_k* curvature_diff)
        control_steer = self.pi_2_pi(stanley_steer)
        control_steer = np.clip(control_steer, -abs_max_steer, abs_max_steer)
        # direct speed control, clip by acceleration
        control_speed = min(path.speed[currentIndex], np.sqrt(comfort_acc / max(np.abs(curvature), 0.0001)))
        print("control_steer = ", control_steer)
        return control_steer, control_speed, psi_diff, Distance
    
    def speed_controller(self, v_ref, v, kp=1.5, max_acc=3.0, max_dec=-0.4):
        a = kp * (v_ref - v)
        a = np.clip(a, max_dec, max_acc)

        return a

    def find_nearest_point(self, x, y, path):
        # 计算每个参考点到 (x, y) 的距离
        if len(path.x) == 0:
            raise ValueError("path is empty")
        distances = np.sqrt((path.x - x)**2 + (path.y - y)**2)
        # 找到最近的参考点的索引
        nearest_index = np.argmin(distances)
        return nearest_index
    
    def find_lookahead_point(self, x, y, path, lookahead_distance):
        if len(path.x) == 0:
            raise ValueError("path is empty")
        
        # 计算每个参考点到 (x, y) 的距离
        distances = np.sqrt((path.x - x)**2 + (path.y - y)**2)
        # 找到最近的参考点的索引
        nearest_index = np.argmin(distances)
        
        # 从最近点开始累积
        cumulative_dist = 0.0
        for i in range(nearest_index, len(path.x)-1):
            dx = path.x[i + 1] - path.x[i]
            dy = path.y[i + 1] - path.y[i]
            segment = np.sqrt(dx ** 2 + dy ** 2)
            cumulative_dist += segment
            if cumulative_dist > lookahead_distance:
                nearest_index = i
        return nearest_index
    
    def kinematics_model(self, x, y, yaw, v, 
                        steer, a, wheelBase=2.6, delta_t=0.01):
        """
        motion model: rear-wheel drive kinematic model
        """
        old_state = np.array([float(x), float(y), float(yaw)])
        v = float(v)
        steer = float(steer)
        stateDot = np.array([v * np.cos(yaw), v * np.sin(yaw), v / wheelBase * np.tan(steer)])
        new_state = old_state + stateDot * delta_t
        new_state[2] = self.pi_2_pi(new_state[2])
        
        new_v = float(v) + float(a) * delta_t
        return new_state[0], new_state[1], new_state[2], new_v
    
    def dynamic_model(self, x, y, yaw, vx, vy, r,
                        steer, ax,
                        wheelBase = 2.6, delta_t=0.01):
        # ===== Tesla Model 3 parameters =====
        m = 1850.0
        Iz = 2875.0

        lf = 1.20
        lr = 1.68

        Cf = 80000.0
        Cr = 80000.0

        # avoid division by zero
        vx = max(vx, 0.1)

        # slip angles
        alpha_f = steer - (vy + lf * r) / vx
        alpha_r = -(vy - lr * r) / vx

        # tire lateral forces
        Fyf = Cf * alpha_f
        Fyr = Cr * alpha_r

        # vehicle dynamics
        vx_dot = ax

        vy_dot = (Fyf + Fyr) / m - vx * r

        r_dot = (lf * Fyf - lr * Fyr) / Iz

        # state update
        vx += vx_dot * delta_t
        vy += vy_dot * delta_t
        r += r_dot * delta_t

        yaw += r * delta_t
        yaw = self.pi_2_pi(yaw)

        x += (vx * np.cos(yaw) - vy * np.sin(yaw)) * delta_t
        y += (vx * np.sin(yaw) + vy * np.cos(yaw)) * delta_t

        return x, y, yaw, vx, vy, r
    
    def pi_2_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def calrealpsi(self, psi_1, psi_2):
        psi_1 = self.pi_2_pi(psi_1)
        psi_2 = self.pi_2_pi(psi_2)
        if psi_1 - psi_2 > np.pi:
            realpsi = self.pi_2_pi(psi_1 - 2 * np.pi - psi_2)
        elif psi_1 - psi_2 < -np.pi:
            realpsi = self.pi_2_pi(psi_1 + 2 * np.pi - psi_2)
        else:
            realpsi = psi_1 - psi_2
        return realpsi
    
    def generate_transition_trajectory(self, ego_x, ego_y, ego_speed, ego_yaw, ego_a, ego_kappa):
        path = self.generate_trajectory(ego_x, ego_y, ego_speed, ego_yaw, ego_a, ego_kappa, True, self.planner_param_init, target_speed=30.0 / 3.6)
        return path
    
    def debug_sim_simulation(self, stanley_control_flag):
        """
        试验时，车辆的起始位姿如下
        note: 'y' and 'yaw' in carla need to multiply -1
        ego_x = 456.56, 
        ego_y = -345.93
        ego_yaw = 2.79  # rad 
        adj_x = 411.5616
        adj_y = -239.2926
        adj_yaw = 0.0259 # rad

        从弯道开始测试
        ego_x, ego_y, ego_yaw, ego_speed
        (402.1652903166725, -243.412617909122, 0.025934149601362703, 11.11111111111111)
        """
        # # 试验起点
        # ego_x = 456.56
        # ego_y = -345.93
        # ego_speed = 30.0 / 3.6  # current speed [m/s]
        # ego_yaw = 2.79 # rad
        # ego_a = 0.0
        # ego_kappa = 0.0

        # # 定曲率弯道起点
        # ego_x = 381.03
        # ego_y = -318.5
        # ego_speed = 30.0 / 3.6  # current speed [m/s]
        # ego_yaw = 2.79 # rad
        # ego_a = 0.0
        # ego_kappa = 0.0

        # 试验弯道起点 tx[0]
        ego_x = 402.1653
        ego_y = -243.4126
        ego_speed = 35.0 / 3.6  # current speed [m/s]
        ego_yaw = 0.0259 # rad
        ego_a = 0.0
        ego_kappa = 0.0

        # 试验弯道终点 tx[375]本来是idx=380，往前一点
        # ego_x = 569.332
        # ego_y = -303.477
        # ego_speed = 35.0 / 3.6  # current speed [m/s]
        # ego_yaw = -0.724 # rad
        # ego_a = 0.0
        # ego_kappa = 0.0

        plan_ego_x = ego_x
        plan_ego_y = ego_y
        plan_ego_yaw = ego_yaw
        plan_ego_speed = ego_speed
        plan_ego_a = ego_a
        plan_ego_kappa = ego_kappa

        # Store current trigger state for next iteration
        if not hasattr(self, 'prev_trigger'):
            self.prev_trigger = False
        path_transition = None
        path = None
        last_path = None

        if self.show_animation:
            # create one combined figure with 3 subplots (road view, frenet s-l, speed)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            ax0, ax1, ax2 = axs[0], axs[1], axs[2]

            ax0.cla()
            ax0.plot(self.road[:,0], self.road[:,1], 'b')
            ax0.plot(self.road[:,2], self.road[:,3], 'b')
            ax0.plot(self.road[:,4], self.road[:,5], 'b')
            ax0.plot(self.polyplanner.tx, self.polyplanner.ty, 'b--')
            ax0.plot(ego_x, ego_y,'b.')
            ax0.set_xlabel("X[m]")
            ax0.set_ylabel("Y[m]")

            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'

            ax1.cla()
            ax1.set_facecolor("grey")
            ax1.plot(np.linspace(0, 600, 100), np.ones(100)*1.875, 'k', linewidth=1)
            ax1.plot(np.linspace(0, 600, 100), np.zeros(100), 'k--', linewidth=1)
            ax1.plot(np.linspace(0, 600, 100), np.ones(100)*-1.875, 'k', linewidth=1)
            ax1.plot(np.linspace(0, 600, 100), np.ones(100)*0.8, 'k--', linewidth=1)
            ax1.plot(np.linspace(0, 600, 100), np.ones(100)*-0.8, 'k--', linewidth=1)
            ego_s, ego_s_dot, ego_s_ddot, ego_l, ego_l_dot, ego_l_ddot = self.polyplanner.calculate_frenet_coordinates(ego_x, ego_y, ego_yaw, ego_speed, ego_kappa, ego_a)
            ax1.plot(ego_s, ego_l, marker='.', color='b')
            ax1.set_xlabel("s[m]")
            ax1.set_ylabel("l[m]")
            ax1.grid(True)

            ax2.cla()
            ax2.plot(0, ego_speed, marker='.', color='b')
            ax2.set_ylim(5,30)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Speed[m/s]")

            # single key handler for the combined figure
            fig.canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        # simulation parameters
        plan_dt = 0.1      # 10 Hz
        control_dt = 0.01  # 100 Hz
        poly_dt = 0.01     # NOTE check that polyplanner内部点与点间隔0.01秒
        plan_steps = int(plan_dt / control_dt) # 10
        road_refresh_steps = plan_steps - 2
        sim_time = 70.0 # 5 minutes
        control_steps = int(sim_time / control_dt)
        
        # ===== 统计变量 =====
        lateral_error_list = []
        yaw_error_list = []
        speed_error_list = []
        plan_time = 0

        for step in range(control_steps):
            # ======================
            # 10 Hz 规划
            # ======================
            if step % plan_steps == 0:
                current_trigger = self.env_data.curve_in_check(ego_x, ego_y)
                if current_trigger:
                    if plan_time == 0:
                        plan_ego_x = ego_x
                        plan_ego_y = ego_y
                        plan_ego_yaw = ego_yaw
                        plan_ego_speed = ego_speed
                        plan_ego_a = ego_a
                        plan_ego_kappa = ego_kappa
                    elif last_path is not None and len(last_path.x) > plan_steps:
                    # 选取上一帧轨迹中起始时间往后 0.1s 的点，因为规划周期0.1s
                    # polyplanner内部点与点间隔0.01秒，所以对应索引为 0.1/0.01 = 10，即 plan_steps
                        idx = int(plan_dt / poly_dt)  
                        plan_ego_x = last_path.x[idx]
                        plan_ego_y = last_path.y[idx]
                        plan_ego_yaw = last_path.yaw[idx]
                        plan_ego_speed = last_path.speed[idx]
                        plan_ego_a = last_path.a[idx]
                        plan_ego_kappa = last_path.c[idx]
                        if np.hypot(plan_ego_x - ego_x, plan_ego_y - ego_y) > 1.0:
                            print("Warning: planned point is far from actual position, using current state for planning")
                            plan_ego_x = ego_x
                            plan_ego_y = ego_y
                            plan_ego_yaw = ego_yaw
                            plan_ego_speed = ego_speed
                            plan_ego_a = ego_a
                            plan_ego_kappa = ego_kappa
                            
                # 出弯瞬间生成过渡轨迹
                if self.prev_trigger and not current_trigger:
                    plan_ego_x = ego_x
                    plan_ego_y = ego_y
                    plan_ego_yaw = ego_yaw
                    plan_ego_speed = ego_speed
                    plan_ego_a = ego_a
                    plan_ego_kappa = ego_kappa

                    path_transition = self.generate_transition_trajectory(plan_ego_x, plan_ego_y, plan_ego_speed, plan_ego_yaw, plan_ego_a, plan_ego_kappa)
                    if self.show_animation:
                        ax0.plot(path_transition.x, path_transition.y,'k')
                
                if path_transition is None:
                    path = self.generate_trajectory(plan_ego_x, plan_ego_y, plan_ego_speed, plan_ego_yaw, plan_ego_a, plan_ego_kappa, current_trigger, self.planner_param_init)
                    if current_trigger:
                        plan_time += 1
                        # 更新 last_path 供下一轮使用
                        last_path = copy.deepcopy(path)
                    # plt.figure(0)
                    # plt.plot(path.x, path.y, marker='.', markersize=2, color='lightgray')
                
                ref_path = copy.deepcopy(path_transition) if path_transition else copy.deepcopy(path)

                if self.show_animation:
                    ax2.plot(step + np.arange(len(ref_path.speed)), ref_path.speed, color='b')

                self.prev_trigger = current_trigger

            # if (not current_trigger) and step % road_refresh_steps == 0:
            if not current_trigger:
                plan_ego_x = ego_x  
                plan_ego_y = ego_y
                plan_ego_yaw = ego_yaw
                plan_ego_speed = ego_speed
                plan_ego_a = ego_a
                plan_ego_kappa = ego_kappa

            # ======================
            # 100 Hz 控制
            # ======================
            if stanley_control_flag == True:
                st_angle, target_v, psi_diff, ey = self.stanley_controller(ego_x, ego_y, ego_yaw, ego_speed, ref_path, wheelBase=2.6)
                accelerate = self.speed_controller(target_v, ego_speed)
                # ======================
                # 运动学更新
                # ======================
                prev_speed = ego_speed
                ego_x, ego_y, ego_yaw, ego_speed = self.kinematics_model(ego_x, ego_y, ego_yaw, ego_speed, st_angle, accelerate, wheelBase=2.6, delta_t=control_dt)

                # ======================
                # 方法1.更新加速度 、kappa, kappa = omega / v = v/L*tan(delta)/v = tan(delta)/L
                # 方法2.更新加速度、kappa：始终用参考路径在当前位置的曲率，与 else 分支一致
                # ======================
                ego_a = (ego_speed - prev_speed) / control_dt
                # ego_a = 0.0
                # # 方法1
                # ego_kappa = np.tan(st_angle) / 2.6      # wheelBase = 2.6
                # 方法2
                print("ego_kappa = ", ego_kappa)
                print("ego_yaw = ", ego_yaw)
                path_test = copy.deepcopy(ref_path)
                path_test.x = self.polyplanner.tx
                path_test.y = self.polyplanner.ty
                path_test.yaw = self.polyplanner.tyaw
                path_test.c = self.polyplanner.tc
                nearest_idx = self.find_nearest_point(ego_x, ego_y, path_test)
                if len(path_test.c) > 0:
                    nearest_idx = min(nearest_idx, len(path_test.c) - 1)
                    ego_kappa = path_test.c[nearest_idx]
                    # if current_trigger:
                    #     ego_yaw = float(path_test.yaw[nearest_idx])
                    #     ego_x = float(path_test.x[nearest_idx])
                    #     ego_y = float(path_test.y[nearest_idx])
                    print("ref kappa = ", self.polyplanner.tc[nearest_idx])
                    print("ref yaw = ", self.polyplanner.tyaw[nearest_idx])
                else:
                    ego_kappa = 0.0
                
                # ego_ay = ego_speed**2 * ego_kappa     # ay = v^2*kappa
            else:
                ego_x = float(ref_path.x[3 + (step % plan_steps)])
                ego_y = float(ref_path.y[3 + (step % plan_steps)])
                ego_speed = float(ref_path.speed[3 + (step % plan_steps)])
                ego_yaw = float(ref_path.yaw[3 + (step % plan_steps)])
                ego_a = float(ref_path.a[3 + (step % plan_steps)])
                ego_kappa = float(ref_path.c[3 + (step % plan_steps)])

            # ====================
            # Frenet 误差计算
            # ====================
            # ego_s, ego_s_dot, ego_s_ddot, ego_l, ego_l_dot, ego_l_ddot = self.polyplanner.calculate_frenet_coordinates(ego_x, ego_y, ego_yaw, ego_speed, ego_kappa, ego_a)
            # lateral_error_list.append(ego_l)
            # yaw_error_list.append(psi_diff)
            # speed_error_list.append(ego_speed - target_v)
            wheelBase = 2.6
            front_x = ego_x + wheelBase * np.cos(ego_yaw)  
            front_y = ego_y + wheelBase * np.sin(ego_yaw)
            front_ego_s, front_ego_s_dot, front_ego_s_ddot, front_ego_l, front_ego_l_dot, front_ego_l_ddot = self.polyplanner.calculate_frenet_coordinates(front_x, front_y, ego_yaw, ego_speed, ego_kappa, ego_a)
            lateral_error_list.append(front_ego_l)
            yaw_error_list.append(psi_diff)
            speed_error_list.append(ego_speed - target_v)

            # ======================
            # 过渡结束判定
            # ======================
            if path_transition and abs(ego_l) < 0.05:
                    path_transition = None

            if self.show_animation:
                # ax1.plot(ego_s, ego_l, marker='.', color='r')
                ax1.plot(front_ego_s, front_ego_l, marker='.', color='g')
                ax2.plot(step, ego_speed, marker='.', color='r')
                ax0.plot(ego_x, ego_y, marker='.', color='r')
                ax0.set_title("v[km/h]:" + str(ego_speed * 3.6)[0:4])
                fig.canvas.draw_idle()
                plt.pause(0.0001)

        print("Simulation Finish!")
        if self.show_animation:
            try:
                fig.savefig("./figures/wholeplanner/Global_combined.png")
            except Exception:
                pass
        lateral_rms = np.sqrt(np.mean(np.array(lateral_error_list)**2))
        yaw_rms = np.sqrt(np.mean(np.array(yaw_error_list)**2))
        speed_rms = np.sqrt(np.mean(np.array(speed_error_list)**2))
        max_l = np.max(np.array(lateral_error_list))
        print("====== Closed-loop RMS Results ======")
        print("Lateral RMS [m]:", lateral_rms)
        print("Yaw RMS [rad]:", yaw_rms)
        print("Speed RMS [m/s]:", speed_rms)
        print("Max lateral [m]:", max_l)

        plt.show()

if __name__ == '__main__':
    road_env = natural_road_load()
    planner = GlobalPlanner(road_env, lane_id=1)
    planner.debug_sim_simulation(stanley_control_flag=True)
