"""
Stanley Lateral Controller.

Implements the Stanley method for lateral path tracking, originally
described in:

    Thrun, S. et al. (2006). Stanley: The robot that won the DARPA
    Grand Challenge. *Journal of Field Robotics*, 23(9), 661–692.

The controller computes a steering angle command that drives the
vehicle to follow a pre-computed reference path by minimising:

1. **Heading error** – the difference between the vehicle's current
   heading and the heading of the nearest path segment.
2. **Cross-track error** – the lateral distance from the front axle
   to the nearest point on the path.

Typical usage
-------------
>>> import numpy as np
>>> from trajectory_planner.stanley_controller import StanleyController
>>> cx = np.linspace(0, 50, 100)
>>> cy = np.zeros(100)
>>> cyaw = np.zeros(100)
>>> controller = StanleyController(k=0.5, max_steer=np.radians(30))
>>> steer, idx = controller.compute_steering(
...     x=0.0, y=0.5, yaw=0.0, v=2.0, cx=cx, cy=cy, cyaw=cyaw
... )
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class StanleyController:
    """Stanley lateral controller.

    Parameters
    ----------
    k : float
        Gain for the cross-track error term.  Higher values make the
        vehicle correct lateral errors more aggressively.  Default 0.5.
    k_soft : float
        Softening constant added to the vehicle speed in the denominator
        of the cross-track error term to avoid division by zero at low
        speeds (m/s).  Default 1.0.
    max_steer : float
        Maximum steering angle magnitude (radians).  Default π/4 (45°).
    wheelbase : float
        Distance between front and rear axles (m).  Used to map the
        front-axle position from the vehicle's centre/rear position.
        Set to 0 if *x*, *y* already describe the front axle.
        Default 2.9.
    """

    def __init__(
        self,
        k: float = 0.5,
        k_soft: float = 1.0,
        max_steer: float = math.pi / 4,
        wheelbase: float = 2.9,
    ) -> None:
        self.k = k
        self.k_soft = k_soft
        self.max_steer = max_steer
        self.wheelbase = wheelbase

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_steering(
        self,
        x: float,
        y: float,
        yaw: float,
        v: float,
        cx: np.ndarray,
        cy: np.ndarray,
        cyaw: np.ndarray,
        last_target_idx: int = 0,
    ) -> Tuple[float, int]:
        """Compute the Stanley steering command.

        Parameters
        ----------
        x, y : float
            Current position of the vehicle's rear axle (m).
        yaw : float
            Current heading of the vehicle (rad).
        v : float
            Current longitudinal speed (m/s).
        cx, cy : array-like of float
            Path waypoint positions (m).
        cyaw : array-like of float
            Path waypoint headings (rad).
        last_target_idx : int
            Index of the previously targeted waypoint; used as a lower
            bound so the controller never tracks backwards.

        Returns
        -------
        steer : float
            Steering angle command (rad), clamped to ±``max_steer``.
        target_idx : int
            Index of the nearest waypoint on the path.
        """
        cx = np.asarray(cx, dtype=float)
        cy = np.asarray(cy, dtype=float)
        cyaw = np.asarray(cyaw, dtype=float)

        # Front-axle position
        fx = x + self.wheelbase * math.cos(yaw)
        fy = y + self.wheelbase * math.sin(yaw)

        # Find the nearest waypoint (no backtracking)
        target_idx = self._nearest_waypoint_index(fx, fy, cx, cy, last_target_idx)

        # Heading error
        theta_e = self._normalise_angle(cyaw[target_idx] - yaw)

        # Cross-track error (signed lateral distance to nearest point).
        # Positive when the front axle is to the right of the path direction,
        # which requires a left (positive) steering correction.
        dx = fx - cx[target_idx]
        dy = fy - cy[target_idx]
        cross_track_error = math.copysign(
            math.hypot(dx, dy),
            math.sin(cyaw[target_idx]) * dx - math.cos(cyaw[target_idx]) * dy,
        )

        # Stanley formula
        theta_d = math.atan2(
            self.k * cross_track_error, self.k_soft + abs(v)
        )
        steer = theta_e + theta_d
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        return steer, target_idx

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_waypoint_index(
        fx: float,
        fy: float,
        cx: np.ndarray,
        cy: np.ndarray,
        last_idx: int,
    ) -> int:
        """Return the index of the waypoint closest to (*fx*, *fy*)."""
        distances = np.hypot(cx[last_idx:] - fx, cy[last_idx:] - fy)
        return int(np.argmin(distances)) + last_idx

    @staticmethod
    def _normalise_angle(angle: float) -> float:
        """Wrap *angle* to the interval (−π, π]."""
        angle = math.fmod(angle, 2.0 * math.pi)
        if angle > math.pi:
            angle -= 2.0 * math.pi
        elif angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle
