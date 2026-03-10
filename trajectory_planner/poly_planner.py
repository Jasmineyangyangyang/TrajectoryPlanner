"""
Quintic Polynomial Trajectory Planner.

Generates smooth, time-parameterised trajectories by fitting a 5th-order
(quintic) polynomial between start and goal states.  Each axis is planned
independently so the planner can be used in 1-D, 2-D or 3-D.

Typical usage
-------------
>>> from trajectory_planner.poly_planner import PolyPlanner
>>> planner = PolyPlanner(max_accel=1.0, max_jerk=0.5)
>>> result = planner.plan(
...     sx=0.0, sy=0.0, syaw=0.0, sv=0.0, sa=0.0,
...     gx=10.0, gy=5.0, gyaw=0.0, gv=0.0, ga=0.0,
... )
>>> result.x   # list of x positions
>>> result.y   # list of y positions
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


class QuinticPolynomial:
    """One-dimensional quintic (5th-order) polynomial.

    The polynomial is defined over *t* ∈ [0, T] and satisfies the boundary
    conditions supplied at construction time:

    - p(0)  = xs,  p(T)  = xe
    - p'(0) = vs,  p'(T) = ve
    - p''(0)= as_,  p''(T)= ae

    Parameters
    ----------
    xs : float
        Start position.
    vs : float
        Start velocity.
    as_ : float
        Start acceleration.
    xe : float
        End position.
    ve : float
        End velocity.
    ae : float
        End acceleration.
    T : float
        Duration (must be > 0).
    """

    def __init__(
        self,
        xs: float,
        vs: float,
        as_: float,
        xe: float,
        ve: float,
        ae: float,
        T: float,
    ) -> None:
        if T <= 0:
            raise ValueError(f"Duration T must be positive, got {T}")

        self.a0 = xs
        self.a1 = vs
        self.a2 = as_ / 2.0

        A = np.array(
            [
                [T**3, T**4, T**5],
                [3 * T**2, 4 * T**3, 5 * T**4],
                [6 * T, 12 * T**2, 20 * T**3],
            ]
        )
        b = np.array(
            [
                xe - self.a0 - self.a1 * T - self.a2 * T**2,
                ve - self.a1 - 2 * self.a2 * T,
                ae - 2 * self.a2,
            ]
        )
        coef = np.linalg.solve(A, b)
        self.a3, self.a4, self.a5 = coef[0], coef[1], coef[2]

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def position(self, t: float) -> float:
        """Position at time *t*."""
        return (
            self.a0
            + self.a1 * t
            + self.a2 * t**2
            + self.a3 * t**3
            + self.a4 * t**4
            + self.a5 * t**5
        )

    def velocity(self, t: float) -> float:
        """First derivative (velocity) at time *t*."""
        return (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * t**2
            + 4 * self.a4 * t**3
            + 5 * self.a5 * t**4
        )

    def acceleration(self, t: float) -> float:
        """Second derivative (acceleration) at time *t*."""
        return (
            2 * self.a2
            + 6 * self.a3 * t
            + 12 * self.a4 * t**2
            + 20 * self.a5 * t**3
        )

    def jerk(self, t: float) -> float:
        """Third derivative (jerk) at time *t*."""
        return (
            6 * self.a3
            + 24 * self.a4 * t
            + 60 * self.a5 * t**2
        )


@dataclass
class TrajectoryResult:
    """Container for the output of :class:`PolyPlanner`.

    Attributes
    ----------
    t : List[float]
        Time stamps.
    x, y : List[float]
        Positions in the world frame.
    yaw : List[float]
        Heading angle (radians) at each time step.
    v : List[float]
        Speed (m/s).
    a : List[float]
        Longitudinal acceleration (m/s²).
    jerk : List[float]
        Longitudinal jerk (m/s³).
    """

    t: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    a: List[float] = field(default_factory=list)
    jerk: List[float] = field(default_factory=list)


class PolyPlanner:
    """Quintic polynomial trajectory planner in 2-D.

    Plans a trajectory from a start state to a goal state by fitting
    independent quintic polynomials along *x* and *y* and searching
    over candidate time horizons to find the shortest feasible plan
    that satisfies the acceleration and jerk limits.

    Parameters
    ----------
    max_accel : float
        Maximum allowed acceleration magnitude (m/s²).  Default 1.0.
    max_jerk : float
        Maximum allowed jerk magnitude (m/s³).  Default 0.5.
    dt : float
        Time resolution of the output trajectory (s).  Default 0.1.
    min_T : float
        Lower bound of the time-horizon search range (s).  Default 5.0.
    max_T : float
        Upper bound of the time-horizon search range (s).  Default 100.0.
    d_T : float
        Step size of the time-horizon search range (s).  Default 5.0.
    """

    def __init__(
        self,
        max_accel: float = 1.0,
        max_jerk: float = 0.5,
        dt: float = 0.1,
        min_T: float = 5.0,
        max_T: float = 100.0,
        d_T: float = 5.0,
    ) -> None:
        self.max_accel = max_accel
        self.max_jerk = max_jerk
        self.dt = dt
        self.min_T = min_T
        self.max_T = max_T
        self.d_T = d_T

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        sx: float,
        sy: float,
        syaw: float,
        sv: float,
        sa: float,
        gx: float,
        gy: float,
        gyaw: float,
        gv: float,
        ga: float,
    ) -> Optional[TrajectoryResult]:
        """Plan a trajectory from start to goal.

        Parameters
        ----------
        sx, sy : float
            Start position (m).
        syaw : float
            Start heading (rad).
        sv : float
            Start speed (m/s).
        sa : float
            Start acceleration (m/s²).
        gx, gy : float
            Goal position (m).
        gyaw : float
            Goal heading (rad).
        gv : float
            Goal speed (m/s).
        ga : float
            Goal acceleration (m/s²).

        Returns
        -------
        TrajectoryResult or None
            The planned trajectory, or *None* if no feasible plan was found
            within the configured time-horizon range.
        """
        vxs = sv * math.cos(syaw)
        vys = sv * math.sin(syaw)
        axs = sa * math.cos(syaw)
        ays = sa * math.sin(syaw)

        vxg = gv * math.cos(gyaw)
        vyg = gv * math.sin(gyaw)
        axg = ga * math.cos(gyaw)
        ayg = ga * math.sin(gyaw)

        T = self.min_T
        while T <= self.max_T:
            xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
            yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)

            result = self._sample(xqp, yqp, T)
            if self._is_feasible(result):
                return result

            T += self.d_T

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample(
        self, xqp: QuinticPolynomial, yqp: QuinticPolynomial, T: float
    ) -> TrajectoryResult:
        result = TrajectoryResult()
        t = 0.0
        while t <= T + 1e-9:
            result.t.append(t)
            result.x.append(xqp.position(t))
            result.y.append(yqp.position(t))

            vx = xqp.velocity(t)
            vy = yqp.velocity(t)
            result.yaw.append(math.atan2(vy, vx))
            result.v.append(math.hypot(vx, vy))

            ax = xqp.acceleration(t)
            ay = yqp.acceleration(t)
            result.a.append(math.hypot(ax, ay))

            jx = xqp.jerk(t)
            jy = yqp.jerk(t)
            result.jerk.append(math.hypot(jx, jy))

            t += self.dt

        return result

    def _is_feasible(self, result: TrajectoryResult) -> bool:
        if any(abs(a) > self.max_accel for a in result.a):
            return False
        if any(abs(j) > self.max_jerk for j in result.jerk):
            return False
        return True
