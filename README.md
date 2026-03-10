### 1. 全局规划器 (globalplanner.py)
#### 1.1模块概述
`globalplanner.py` 实现了车辆的全局路径规划与跟踪控制功能，是自动驾驶系统中的核心规划模块。该模块集成了道路规划、多项式轨迹规划和Stanley控制器，负责生成平滑、安全、可执行的行驶轨迹。

### 2. 全局路径点文件 (global.csv)

#### 2.1 文件位置
`./global.csv`

#### 2.2 文件说明
`global.csv` 是包含道路全局路径点的核心数据文件，用于定义车辆需要遵循的全局行驶路径。\
双车道匝道，每条车道宽 3.75m \
每个路点间隔 0.5m

#### 2.3 数据格式
该文件为标准的CSV（逗号分隔值）格式，包含以下字段：

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| `center_x` | float | 路径点的X坐标（单位：米） |
| `center_y` | float | 路径点的Y坐标（单位：米） |
| `left_x` | float | 左边界路径点的X坐标（单位：米） |
| `left_y` | float | 左边界路径点的Y坐标（单位：米） |
| `right_x` | float | 右边界路径点的X坐标（单位：米） |
| `right_y` | float | 右边界路径点的Y坐标（单位：米） |
| `left_center_x` | float | 左道路中心线路径点的X坐标（单位：米） |
| `left_center_y` | float | 左道路中心线路径点的Y坐标（单位：米） |
| `right_center_x` | float | 右道路中心线路径点的X坐标（单位：米） |
| `right_center_y` | float | 右道路中心线路径点的Y坐标（单位：米） |
| `speed` | float | 建议行驶速度（单位：m/s） |
| `rightcenter_cur` | float | 右弯道曲率（单位：m^-1）这里右转曲率为正，与业内常规：左正右负不同，所以后续调用乘以-1 |
| `leftcenter_cur` | float | 左弯道曲率（单位：m^-1）这里右转曲率为正，与业内常规：左正右负不同，所以后续调用乘以-1|
| `in_bend` | int | 路径点是否在试验弯道内flag |

#### 2.4 数据示例
```csv
center_x,center_y,left_x,left_y,right_x,right_y,left_center_x,left_center_y,right_center_x,right_center_y,road_speed,rightcenter_cur,leftcenter_cur,in_bend
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.8,0.0,0.0,1
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,10.0,0.0,0.0,1
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,10.0,0.0,0.0,1
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,10.0,0.0,0.0,1