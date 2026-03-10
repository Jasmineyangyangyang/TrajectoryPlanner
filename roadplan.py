import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import pathlib
current_dir = pathlib.Path(os.getcwd())
sys.path.append(current_dir)

# for debug
from global_road import natural_road_load

class RoadPath:
    def __init__(self):
        self.t = []     # time
        self.x = []     # global x position
        self.y = []     # global y position
        self.speed = [] # global speed
        self.a = []     # global accelearion
        self.c = []     # global curvature

class Roadplanner():
    def __init__(self, env_data, lane_id, preview_time=5.0):
        self.env_data = env_data
        # self.road = self.env_data.read_from_csv('G:\\Jiaxin_OneDrive\\OneDrive\\桌面\\carla_road\\IRL_poly_newRP\\IRL_env\\envs\\trajectory_planner\\polytest241110\\csvdata')
        self.road = self.env_data.read_from_csv('./')
        self.road_center = []   # this is the Cartesian center of the lane which ego vehicle is driving.
        self.road_left = self.road[:,0:2]     # [x, y]
        self.road_right = self.road[:,2:4]    # [x, y]
        self.road_curvature = []
        for i in range(len(self.road)):
            if lane_id == 0:     # outside
                center_left_x = self.road[i][0]
                center_left_y = self.road[i][1]
                center_right_x = self.road[i][4]
                center_right_y = self.road[i][5]
                curvature = self.road[i][8] * -1
                
            elif lane_id == 1:   # inside
                center_left_x = self.road[i][4]
                center_left_y = self.road[i][5]
                center_right_x = self.road[i][2]
                center_right_y = self.road[i][3]
                curvature = self.road[i][7] * -1

            road_center_x = (center_left_x + center_right_x) / 2.0
            road_center_y = (center_left_y + center_right_y) / 2.0
            self.road_center.append([road_center_x, road_center_y])
            self.road_curvature.append(curvature)
        self.road_center = np.array(self.road_center)  # Cartesian coordinate
        self.road_curvature = np.array(self.road_curvature)
        self.preview_time = preview_time
        self.wspeed = self.road[:, 6]
        self.wx = self.road_center[:, 0]
        self.wy = self.road_center[:, 1]
        self.c = self.road_curvature
        self.show_animation = True
    
    def find_nearest_point(self, x, y):
        # 计算每个参考点到 (x, y) 的距离
        distances = np.sqrt((self.wx - x)**2 + (self.wy - y)**2)
        # 找到最近的参考点的索引
        nearest_index = np.argmin(distances)
        return nearest_index

    def road_trajectory(self, ego_x, ego_y, ego_speed):
        path = RoadPath()
        start_index = self.find_nearest_point(ego_x, ego_y)
        point_num = math.ceil(self.preview_time * ego_speed / 0.5)
        if start_index + point_num <= self.wx.shape[0]:
            end_index = start_index + point_num
            path.x = self.wx[start_index:end_index]
            path.y = self.wy[start_index:end_index]
            # Calculate yaw angles from path points
            path.yaw = np.zeros(len(path.x)-1)
            for i in range(len(path.x)-1):
                dx = path.x[i+1] - path.x[i]
                dy = path.y[i+1] - path.y[i]
                path.yaw[i] = np.arctan2(dy, dx)
            # Add last yaw angle by copying previous one
            path.yaw = np.append(path.yaw, path.yaw[-1])
            path.speed = self.wspeed[start_index:end_index]
            path.c = self.c[start_index:end_index]
        else:
            end_index = start_index + point_num - self.wx.shape[0]
            path.x = np.concatenate((self.wx[start_index:], self.wx[:end_index]), axis=0)
            path.y = np.concatenate((self.wy[start_index:], self.wy[:end_index]), axis=0)
            path.c = np.concatenate((self.c[start_index:], self.c[:end_index]), axis=0)
            # Calculate yaw angles from path points
            path.yaw = np.zeros(len(path.x)-1)
            for i in range(len(path.x)-1):
                dx = path.x[i+1] - path.x[i]
                dy = path.y[i+1] - path.y[i]
                path.yaw[i] = np.arctan2(dy, dx)
            # Add last yaw angle by copying previous one
            path.yaw = np.append(path.yaw, path.yaw[-1])
            path.speed = np.concatenate((self.wspeed[start_index:], self.wspeed[:end_index]), axis=0)
        path.a = np.zeros(len(path.x))

        return path
    
    # def road_trajectory(self, ego_x, ego_y, ego_speed, ob=np.array([])):
    #     path = RoadPath()
    #     path.x = self.wx
    #     path.y = self.wy
    #     path.s_d = self.wspeed
    #     return path
    
    def debug_sim(self):
        """
        试验时，车辆的起始位姿如下
        note: 'y' and 'yaw' in carla need to multiply -1
        ego_x = 456.56, 
        ego_y = -345.93
        ego_yaw = 2.79 rad 
        adj_x = 411.5616
        adj_y = -239.2926
        adj_yaw = 0.0259 rad
        """
        ego_x = self.road_center[380, 0]
        ego_y = self.road_center[380, 1]
        # dy = self.road_center_trajectory[381, 9] - self.road_center_trajectory[380, 9]
        # dx = self.road_center_trajectory[381, 8] - self.road_center_trajectory[381, 8]
        # ego_yaw = np.arctan2(dy, dx)
        ego_speed = 15.0 / 3.6  # current speed [m/s]
        # ego_accel = 0.0  # current acceleration [m/ss]

        ob = np.array([])

        SIM_LOOP = 5000000 # simulation loop
        
        for i in range(SIM_LOOP):
            path = self.road_trajectory(ego_x, ego_y, ego_speed)

            ego_x = path.x[1]
            ego_y = path.y[1]
            ego_speed = path.speed[1]

            if np.hypot(path.x[1] - self.road_center[-1, 0], path.y[1] - self.road_center[-1, 1]) <= 1.0:
                print("near Bend start")
                break
            elif np.hypot(path.x[1] - self.road_center[380, 0], path.y[1] - self.road_center[380, 1]) <= 1.0:
                print("near Bend end")
                # break

            if self.show_animation:
                plt.cla()
                plt.plot(self.road[:,0], self.road[:,1], 'b')
                plt.plot(self.road[:,2], self.road[:,3], 'b')
                plt.plot(self.road[:,4], self.road[:,5], 'b')
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(self.wx, self.wy)
                if not len(ob) == 0:
                    plt.plot(ob[:, 0], ob[:, 1], "xk")
                plt.plot(path.x[1:], path.y[1:], "-or")
                plt.plot(path.x[1], path.y[1], "vc")
                # plt.xlim(path.x[1] - area, path.x[1] + area)
                # plt.ylim(path.y[1] - area, path.y[1] + area)
                plt.title("v[km/h]:" + str(ego_speed * 3.6)[0:4])
                plt.grid(True)
                plt.pause(0.0001)
        
        print("Finish")
        if self.show_animation:  # pragma: no cover
            plt.grid(True)
            plt.pause(0.0001)
            plt.show()

if __name__ == '__main__':
    env_data = natural_road_load()
    planner = Roadplanner(env_data, lane_id=1)
    planner.debug_sim()