"""
Defines the configuration parameters of road curves and the calculation method for the trigger area boundary box;
Loads road data and constructs road trajectories;
Updates the trigger status of curves and checks if the current position is inside the trigger area of a curve;
Plots the road centerline and curve trigger area.
"""
import csv  # Import the library for reading and writing CSV files
import os   # Import the library for operating system functions
import numpy as np  # Import the library for scientific computing
import matplotlib.pyplot as plt  # Import the library for plotting

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

class CurveFlag():
    """
    Defines the configuration parameters of road curves and calculates the boundary box of the trigger area.
    """
    def __init__(self) -> None:
        # Initialize the curve configuration parameters
        self.triggerbox_width = 10.  # Width of the trigger area
        self.triggerbox_length = 10.  # Length of the trigger area
        
        # The following are the coordinates of the trigger points for different curves
        # Curve (clockwise direction)
        self.cur_in_left = np.array([[402.02, -237.79]])
        self.cur_in_right = np.array([[402.214, -245.289]]) 
        self.cur_out_left = np.array([[574.932, -300.919]])
        self.cur_out_right = np.array([[569.962, -306.537]])

class natural_road_load():
    """
    Loads road data, constructs road trajectories, updates the trigger status of curves, and plots the road centerline and trigger area.
    """
    def __init__(self):   
        # Initialize the natural road load
        self.curvfg = CurveFlag()  # Curve configuration object
                     
    
    def read_from_csv(self, filepath='./'):
        """Reads road information from a CSV file"""
        road_filename = os.path.join(filepath, 'global_road.csv')
        self.road = []
        print("Loading global_road Data...")
        with open(road_filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                left_x = float(row['left_x'])
                left_y = float(row['left_y'])
                right_x = float(row['right_x'])
                right_y = float(row['right_y'])
                center_x = float(row['center_x'])
                center_y = float(row['center_y'])
                road_speed = float(row['road_speed'])
                rightcenter_cur = float(row['rightcenter_cur'])
                leftcenter_cur = float(row['leftcenter_cur'])
                in_bend = float(row['in_bend'])
                self.road.append([left_x, left_y, right_x, right_y, center_x, center_y,\
                                  road_speed, rightcenter_cur, leftcenter_cur, in_bend])
        print("Load global_road successfully!")
        print(f"the road width = {round(np.hypot(self.road[0][2]-self.road[0][4], self.road[0][3] - self.road[0][5]), 2)}")
        self.road_trajectory = np.array(self.road)
        return self.road_trajectory
        
    def curve_in_check(self, x, y):
        """Checks if a point is inside a curve"""
        in_bend_vec = self.curvfg.cur_in_right - self.curvfg.cur_in_left
        out_bend_vec = self.curvfg.cur_out_right - self.curvfg.cur_out_left
        vehile_vec_in = np.array([x, y]) - self.curvfg.cur_in_left
        vehile_vec_out = np.array([x, y]) - self.curvfg.cur_out_left
        if (400.0 <= x <= 576.) and (-308 <= y <= -225):
            if np.cross(vehile_vec_in, in_bend_vec) < 0 and np.cross(vehile_vec_out, out_bend_vec) > 0:
                return True
            else:
                return False
        else:
            return False

    def plot_road(self):
        plt.legend()
        plt.title('Road and Bend Centers')
        plt.plot(self.road_trajectory[:,0], self.road_trajectory[:,1], 'r')
        plt.plot(self.road_trajectory[:,2], self.road_trajectory[:,3], 'r')
        plt.plot(self.road_trajectory[:,4], self.road_trajectory[:,5], 'r--')
        plt.grid(True)
        plt.show()

def wrap_to_pi(theta):
    """
    theta: rad
    turn it into [-pi, pi]
    """
    return np.arctan2(np.sin(theta), np.cos(theta))

# for debug
if __name__ == '__main__':
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
    # Instantiate the natural road load object
    ego_lane_id = 1
    env_data = natural_road_load()
    road = env_data.read_from_csv('./')
    road_center = []   # this is the Cartesian center of the lane which ego vehicle is driving.
    road_left = road[:,0:2]     # [x, y]
    road_right = road[:,2:4]    # [x, y]
    for i in range(len(road)):
        if ego_lane_id == 0:     # outside
            center_left_x = road[i][0]
            center_left_y = road[i][1]
            center_right_x = road[i][4]
            center_right_y = road[i][5]
            
        elif ego_lane_id == 1:   # inside
            center_left_x = road[i][4]
            center_left_y = road[i][5]
            center_right_x = road[i][2]
            center_right_y = road[i][3]

        road_center_x = (center_left_x + center_right_x) / 2.0
        road_center_y = (center_left_y + center_right_y) / 2.0
        road_center.append([road_center_x, road_center_y])
    road_center = np.array(road_center)  # Cartesian coordinate

    curve_flag = []
    
    # Update the trigger status of curves and record them
    for i in range(road_center.shape[0]):
        curve_trigger = env_data.curve_in_check(road_center[i,0], road_center[i,1])
        curve_flag.append([i,curve_trigger])
    
    # check ego's init yaw and adj's init yaw
    if ego_lane_id == 0:
        nearest_index = np.argmin(np.sqrt((road_center[:,0] - 411.5616)**2 + (road_center[:,1] - -239.2926)**2))
        adj_yaw = np.arctan2(road_center[nearest_index+1, 1] - road_center[nearest_index, 1], road_center[nearest_index+1, 0] - road_center[nearest_index, 0])
        adj_yaw = wrap_to_pi(adj_yaw)
        print(f"calculate adj_yaw: {adj_yaw}, give adj_yaw = -0.0259 rad")
    else:
        nearest_index = np.argmin(np.sqrt((road_center[:,0] - 456.56)**2 + (road_center[:,1] - -345.93)**2))
        ego_yaw = np.arctan2(road_center[nearest_index+1, 1] - road_center[nearest_index, 1], road_center[nearest_index+1, 0] - road_center[nearest_index, 0])
        ego_yaw = wrap_to_pi(ego_yaw)
        print(f"calculate ego_yaw: {ego_yaw}, give ego_yaw = 2.79 rad")

    # Plot the road centerline and trigger area
    print(f"ego_lane_id = {ego_lane_id}")
    print(f"bend_start_left_x = {road_left[0,0]}, bend_start_left_y = {road_left[0,1]}")
    print(f"bend_start_right_x = {road_right[0,0]}, bend_start_right_y = {road_right[0,1]}")
    print(f"bend_end_left_x = {road_left[380,0]}, bend_end_left_y = {road_left[380,1]}")
    print(f"bend_end_right_x = {road_right[380,0]}, bend_end_right_y = {road_right[380,1]}")
    
    '''
    plt.figure(231223)
    plt.axis('equal')
    plt.plot(road[:, 0], road[:, 1], 'k', label='road')
    plt.plot(road[:, 2], road[:, 3], 'k')
    plt.plot(road[:, 4], road[:, 5], 'k')
    plt.plot(road_center[:381, 0], road_center[:381, 1], 'k--', label='ego road center')
    # plt.plot(road_center[:381,0], road_center[:381,1], 'bo', markersize=2)
    plt.plot(411.5616, -239.2926, 'co', markersize=2, label='adj_pose')
    plt.plot(402.1653, -243.4126, 'ro', markersize=2, label='ego_pose')
    
    # 在 road_center[380] 处画一条垂直于道路中心线、长度为 8m 的红线
    idx = 380
    if 1 <= idx < road_center.shape[0]:
        # 用相邻点估计切向方向
        if idx == road_center.shape[0] - 1:
            dx = road_center[idx, 0] - road_center[idx - 1, 0]
            dy = road_center[idx, 1] - road_center[idx - 1, 1]
        else:
            dx = road_center[idx + 1, 0] - road_center[idx - 1, 0]
            dy = road_center[idx + 1, 1] - road_center[idx - 1, 1]
        # 法向方向（单位向量）
        norm = np.hypot(dx, dy)
        if norm > 0.0:
            nx = -dy / norm
            ny = dx / norm
            half_len = 6.0  # 一半 4m，总长 8m
            x0, y0 = road_center[idx, 0], road_center[idx, 1]
            x1, y1 = x0 + half_len * nx, y0 + half_len * ny
            x2, y2 = x0 - half_len * nx, y0 - half_len * ny
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label='normal @ idx=380')
    
    idx = 0
    if 0 <= idx < road_center.shape[0]:
        # 用相邻点估计切向方向
        if idx == road_center.shape[0] - 1:
            dx = road_center[idx, 0] - road_center[idx - 1, 0]
            dy = road_center[idx, 1] - road_center[idx - 1, 1]
        elif idx == 0:
            dx = road_center[idx + 1, 0] - road_center[idx, 0]
            dy = road_center[idx + 1, 1] - road_center[idx, 1]
        else:
            dx = road_center[idx + 1, 0] - road_center[idx - 1, 0]
            dy = road_center[idx + 1, 1] - road_center[idx - 1, 1]
        # 法向方向（单位向量）
        norm = np.hypot(dx, dy)
        if norm > 0.0:
            nx = -dy / norm
            ny = dx / norm
            half_len = 6.0  # 一半 4m，总长 8m
            x0, y0 = road_center[idx, 0], road_center[idx, 1]
            x1, y1 = x0 + half_len * nx, y0 + half_len * ny
            x2, y2 = x0 - half_len * nx, y0 - half_len * ny
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label='normal @ idx=0')

    # plt.xlabel('Global X/m')
    # plt.ylabel('Global Y/m')
    # plt.xticks(fontproperties='Times New Roman', size=12)
    # plt.yticks(fontproperties='Times New Roman', size=12)
    # plt.legend(loc='best', fontsize=11)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.savefig('./Figures/global_road.png', dpi=600)  # Save the figure with high resolution
    # plt.show()

    '''

    # ==============================================================================
    # 研究路段ROI分割图
    # ==============================================================================
    # ==============================================================================
    # --- 1. 定义顶刊配色 ---
    color_straight = '#34495E'  # 直线：深邃沥青蓝
    color_clothoid = '#F39C12'  # 回旋线：学术暗金橙
    color_curve    = '#8E44AD'  # 定曲率段：典雅紫藤色

    plt.figure(260330)
    plt.axis('equal')
    plt.plot(road[0:60, 0], road[0:60, 1], color=color_straight, label='道路边界')
    plt.plot(road[60:140, 0], road[60:140, 1], color=color_clothoid)
    plt.plot(road[140:240, 0], road[140:240, 1], color=color_curve)
    plt.plot(road[240:320, 0], road[240:320, 1], color=color_clothoid)
    plt.plot(road[320:380, 0], road[320:380, 1], color=color_straight)
    plt.plot(road[380:405, 0], road[380:405, 1], 'gray')

    plt.plot(road[0:60, 2], road[0:60, 3], color=color_straight)
    plt.plot(road[60:140, 2], road[60:140, 3], color=color_clothoid)
    plt.plot(road[140:240, 2], road[140:240, 3], color=color_curve)
    plt.plot(road[240:320, 2], road[240:320, 3], color=color_clothoid)
    plt.plot(road[320:380, 2], road[320:380, 3], color=color_straight)
    plt.plot(road[380:405, 2], road[380:405, 3], 'gray')

    plt.plot(road[:405, 4], road[:405, 5], 'gray', linestyle='-.', label='道路中心线')
    if ego_lane_id == 0:
        adj_center_x = (road[:,2] + road[:,4]) / 2.0
        adj_center_y = (road[:,3] + road[:,5]) / 2.0
        plt.plot(adj_center_x[17:400], adj_center_y[17:400], '-', linewidth=1.0, label='邻车路径')
    else:
        adj_center_x = (road[:,0] + road[:,4]) / 2.0
        adj_center_y = (road[:,1] + road[:,5]) / 2.0
        plt.plot(adj_center_x[17:400], adj_center_y[17:400], '--', linewidth=1.0, label='邻车路径')
    plt.plot(road_center[:381, 0], road_center[:381, 1], 'k--', linewidth=1.1, label='自车车道中心线')
    plt.plot(411.5616, -239.2926, 'co', markersize=2, label='邻车示意点')
    plt.plot(402.1653, -243.4126, 'ro', markersize=2, label='自车示意点')

    # idx = 380
    idx = 360
    if 1 <= idx < road_center.shape[0]:
        # 用相邻点估计切向方向
        if idx == road_center.shape[0] - 1:
            dx = road_center[idx, 0] - road_center[idx - 1, 0]
            dy = road_center[idx, 1] - road_center[idx - 1, 1]
        else:
            dx = road_center[idx + 1, 0] - road_center[idx - 1, 0]
            dy = road_center[idx + 1, 1] - road_center[idx - 1, 1]
        # 法向方向（单位向量）
        norm = np.hypot(dx, dy)
        if norm > 0.0:
            nx = -dy / norm
            ny = dx / norm
            half_len = 8.0  # 一半 4m，总长 8m
            x0, y0 = road[idx, 4], road[idx, 5]
            x1, y1 = x0 + half_len * nx, y0 + half_len * ny
            x2, y2 = x0 - half_len * nx, y0 - half_len * ny
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=1.0, label='试验弯道截面')
    
    # idx = 0
    idx = 20
    if 0 <= idx < road_center.shape[0]:
        # 用相邻点估计切向方向
        if idx == road_center.shape[0] - 1:
            dx = road_center[idx, 0] - road_center[idx - 1, 0]
            dy = road_center[idx, 1] - road_center[idx - 1, 1]
        elif idx == 0:
            dx = road_center[idx + 1, 0] - road_center[idx, 0]
            dy = road_center[idx + 1, 1] - road_center[idx, 1]
        else:
            dx = road_center[idx + 1, 0] - road_center[idx - 1, 0]
            dy = road_center[idx + 1, 1] - road_center[idx - 1, 1]
        # 法向方向（单位向量）
        norm = np.hypot(dx, dy)
        if norm > 0.0:
            nx = -dy / norm
            ny = dx / norm
            half_len = 8.0  # 一半 4m，总长 8m
            x0, y0 = road[idx, 4], road[idx, 5]
            x1, y1 = x0 + half_len * nx, y0 + half_len * ny
            x2, y2 = x0 - half_len * nx, y0 - half_len * ny
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=1.0)
    
    plt.xlabel('全局 X/m', fontsize=15)
    plt.ylabel('全局 Y/m', fontsize=15)
    plt.legend(loc='best', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() 
    plt.savefig('./Figures/bend_road.png', dpi=600)  # Save the figure with high resolution
    plt.show()

