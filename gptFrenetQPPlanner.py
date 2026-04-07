import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from global_road import natural_road_load


class FrenetQPPlanner:

    def __init__(self):

        self.max_offset = 0.95
        self.n_ctrl = 6

        # cost weights
        self.w_track = 20
        self.w_smooth = 5
        self.w_jerk = 2


    # ------------------------------------------------
    # build Frenet s coordinate
    # ------------------------------------------------

    def compute_s(self, cx, cy):

        s = [0]

        for i in range(1, len(cx)):
            ds = np.hypot(cx[i]-cx[i-1], cy[i]-cy[i-1])
            s.append(s[-1] + ds)

        return np.array(s)


    # ------------------------------------------------
    # QP optimization
    # ------------------------------------------------

    def solve_qp(self, d_ref):

        d = cp.Variable(self.n_ctrl)

        cost = 0

        # track reference trajectory
        cost += self.w_track * cp.sum_squares(d - d_ref)

        # smoothness
        for i in range(self.n_ctrl-1):
            cost += self.w_smooth * cp.square(d[i+1] - d[i])

        # jerk
        for i in range(self.n_ctrl-2):
            cost += self.w_jerk * cp.square(d[i+2] - 2*d[i+1] + d[i])

        constraints = []

        constraints += [d <= self.max_offset]
        constraints += [d >= -self.max_offset]

        prob = cp.Problem(cp.Minimize(cost), constraints)

        prob.solve()

        return d.value


    # ------------------------------------------------
    # generate spline trajectory in Frenet
    # ------------------------------------------------

    def generate_traj(self, s_ctrl, d_ctrl, s_dense):

        spline = CubicSpline(s_ctrl, d_ctrl)
        d_dense = spline(s_dense)

        return d_dense


    # ------------------------------------------------
    # Frenet -> Cartesian
    # ------------------------------------------------

    def frenet_to_xy(self, cx, cy, s_ref, s_query, d_query):

        x = []
        y = []

        for s, d in zip(s_query, d_query):

            idx = np.argmin(np.abs(s_ref - s))

            if idx >= len(cx)-1:
                idx = len(cx)-2

            dx = cx[idx+1] - cx[idx]
            dy = cy[idx+1] - cy[idx]

            yaw = np.arctan2(dy, dx)

            nx = -np.sin(yaw)
            ny = np.cos(yaw)

            x.append(cx[idx] + nx*d)
            y.append(cy[idx] + ny*d)

        return np.array(x), np.array(y)

    # ------------------------------------------------
    # curvature computation
    # ------------------------------------------------

    def compute_curvature(self, x, y):

        dx = np.gradient(x)
        dy = np.gradient(y)

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        kappa = (dx*ddy - dy*ddx) / ((dx**2 + dy**2)**1.5 + 1e-6)

        return np.abs(kappa)


    # ------------------------------------------------
    # speed planning
    # ------------------------------------------------

    def speed_planning(self, kappa):

        ay_max = 3.0      # lateral acceleration limit
        v_max = 15.0      # speed limit

        v_curve = np.sqrt(ay_max / (kappa + 1e-3))

        v = np.minimum(v_curve, v_max)

        return v


    # ------------------------------------------------
    # time parameterization
    # ------------------------------------------------

    def time_parameterization(self, s, v):

        t = np.zeros(len(s))

        for i in range(1, len(s)):

            ds = s[i] - s[i-1]

            t[i] = t[i-1] + ds / max(v[i-1], 0.1)

        return t
    

    # ------------------------------------------------
    # Frenet visualization
    # ------------------------------------------------

    def plot_frenet(self, s_ctrl, d_ctrl, s_dense, d_dense):

        plt.figure(figsize=(8,4))

        plt.scatter(s_ctrl, d_ctrl, color='red', label='control points')

        plt.plot(s_dense, d_dense, linewidth=3, label='trajectory')

        plt.axhline(0, linestyle='--')

        plt.xlabel("s (m)")
        plt.ylabel("d (m)")
        plt.title("Trajectory in Frenet Space")

        plt.grid(True)
        plt.legend()

        plt.show()


    # ------------------------------------------------
    # planning interface
    # ------------------------------------------------

    def plan(self, mode="center", show_frenet=False):

        road = natural_road_load()
        road_data = road.read_from_csv()

        cx = road_data[:,4]
        cy = road_data[:,5]

        s = self.compute_s(cx, cy)

        # curve region (temporary fixed)
        # s_start = s[200]
        # s_end = s[350]
        s_start = s[0]
        s_end = s[380]

        s_ctrl = np.linspace(s_start, s_end, self.n_ctrl)

        # reference trajectories

        if mode == "center":

            d_ref = np.zeros(self.n_ctrl)

        elif mode == "offset":

            d_ref = np.ones(self.n_ctrl) * 0.3

        elif mode == "racing":

            d_ref = np.array([0.8, 0.6, -0.5, -0.5, 0.6, 0.8])

        else:

            raise ValueError("Unknown mode")

        # solve QP
        d_opt = self.solve_qp(d_ref)

        # dense trajectory
        s_dense = np.linspace(s_start, s_end, 500)

        d_dense = self.generate_traj(s_ctrl, d_opt, s_dense)

        # convert to Cartesian
        x_traj, y_traj = self.frenet_to_xy(cx, cy, s, s_dense, d_dense)

        # curvature
        kappa = self.compute_curvature(x_traj, y_traj)
        kappa = gaussian_filter1d(kappa, sigma=2)

        # speed planning
        v = self.speed_planning(kappa)

        # time parameterization
        t = self.time_parameterization(s_dense, v)

        if show_frenet:
            self.plot_frenet(s_ctrl, d_opt, s_dense, d_dense)

        return cx, cy, x_traj, y_traj, s_dense, v, t


# ------------------------------------------------
# main
# ------------------------------------------------

if __name__ == "__main__":

    planner = FrenetQPPlanner()

    cx, cy, x1, y1, s1, v1, t1 = planner.plan("center", True)
    _, _, x2, y2, s2, v2, t2 = planner.plan("offset", True)
    _, _, x3, y3, s3, v3, t3 = planner.plan("racing", True)

    plt.figure(figsize=(8,4))

    plt.plot(s1, v1, label="center")
    plt.plot(s2, v2, label="offset")
    plt.plot(s3, v3, label="racing")

    plt.xlabel("s (m)")
    plt.ylabel("speed (m/s)")
    plt.title("Speed Profile")

    plt.grid(True)
    plt.legend()

    plt.show()