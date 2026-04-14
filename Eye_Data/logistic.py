import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 配置符合要求的绘图参数 =================
plt.rcParams['font.family'] = 'sans-serif'
# 优先使用宋体，如果没有则使用系统备选
plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC', 'STSong'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 14

# 设置数学公式字体为 Times New Roman
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

# ================= 2. 定义函数与生成数据 =================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成从 -10 到 10 的 1000 个点
x = np.linspace(-10, 10, 1000)
y = sigmoid(x)

# ================= 3. 执行绘图 =================
plt.figure(figsize=(8, 5))

# 绘制 Sigmoid 曲线
plt.plot(x, y, color='black', linewidth=2)

# 绘制辅助线：y=0.5 决策边界和 x=0 中线
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

# 设置标签：使用 LaTeX 渲染希腊字母与向量符号
# \Delta 表示特征差值，\mathbf{E} 表示特征向量（加粗）
plt.xlabel(r'$\Delta \mathbf{E}$', fontsize=18, fontname='Times New Roman')
plt.ylabel(r'$y$', fontsize=18, rotation=0, labelpad=15, fontname='Times New Roman')

# 完善细节
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(-0.05, 1.05)
plt.xlim(-10, 10)

# 保存图片
plt.savefig('./Eye_data/Sigmoid_Plot.png', dpi=600, bbox_inches='tight')
plt.show()