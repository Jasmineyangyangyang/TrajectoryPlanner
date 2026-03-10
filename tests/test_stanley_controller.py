"""Unit tests for StanleyController."""

import math
import pytest
import numpy as np

from trajectory_planner.stanley_controller import StanleyController


class TestStanleyController:
    """Tests for StanleyController."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _straight_path(length=50.0, n=100):
        cx = np.linspace(0, length, n)
        cy = np.zeros(n)
        cyaw = np.zeros(n)
        return cx, cy, cyaw

    # ------------------------------------------------------------------
    # Steering output range
    # ------------------------------------------------------------------

    def test_steering_within_max(self):
        controller = StanleyController(k=1.0, max_steer=math.radians(30))
        cx, cy, cyaw = self._straight_path()
        steer, _ = controller.compute_steering(
            x=0.0, y=5.0, yaw=0.0, v=5.0, cx=cx, cy=cy, cyaw=cyaw
        )
        assert abs(steer) <= math.radians(30) + 1e-9

    def test_steering_clamped_large_error(self):
        max_steer = math.radians(20)
        controller = StanleyController(k=100.0, max_steer=max_steer)
        cx, cy, cyaw = self._straight_path()
        steer, _ = controller.compute_steering(
            x=0.0, y=500.0, yaw=0.0, v=1.0, cx=cx, cy=cy, cyaw=cyaw
        )
        assert abs(steer) <= max_steer + 1e-9

    # ------------------------------------------------------------------
    # Perfect tracking
    # ------------------------------------------------------------------

    def test_zero_error_zero_steering(self):
        controller = StanleyController(k=0.5, max_steer=math.radians(30), wheelbase=0.0)
        cx, cy, cyaw = self._straight_path()
        steer, _ = controller.compute_steering(
            x=0.0, y=0.0, yaw=0.0, v=5.0, cx=cx, cy=cy, cyaw=cyaw
        )
        assert steer == pytest.approx(0.0, abs=1e-6)

    # ------------------------------------------------------------------
    # Correct steering direction
    # ------------------------------------------------------------------

    def test_positive_cross_track_steers_right(self):
        """Vehicle to the LEFT of the path (positive y with path heading right).

        Cross-track error is negative (vehicle is not to the right of path),
        so the Stanley correction term theta_d is negative, producing a
        negative (rightward) steering command.
        """
        controller = StanleyController(k=1.0, max_steer=math.radians(45), wheelbase=0.0)
        cx, cy, cyaw = self._straight_path()
        steer, _ = controller.compute_steering(
            x=0.0, y=1.0, yaw=0.0, v=5.0, cx=cx, cy=cy, cyaw=cyaw
        )
        # Negative steer = turn right (clockwise), correcting leftward offset
        assert steer < 0.0

    def test_negative_cross_track_steers_left(self):
        """Vehicle to the RIGHT of the path (negative y with path heading right).

        Cross-track error is positive (vehicle is to the right of path),
        so the Stanley correction term theta_d is positive, producing a
        positive (leftward) steering command.
        """
        controller = StanleyController(k=1.0, max_steer=math.radians(45), wheelbase=0.0)
        cx, cy, cyaw = self._straight_path()
        steer, _ = controller.compute_steering(
            x=0.0, y=-1.0, yaw=0.0, v=5.0, cx=cx, cy=cy, cyaw=cyaw
        )
        # Positive steer = turn left (counter-clockwise), correcting rightward offset
        assert steer > 0.0

    # ------------------------------------------------------------------
    # Target index
    # ------------------------------------------------------------------

    def test_target_idx_non_negative(self):
        controller = StanleyController()
        cx, cy, cyaw = self._straight_path()
        _, idx = controller.compute_steering(
            x=5.0, y=0.0, yaw=0.0, v=2.0, cx=cx, cy=cy, cyaw=cyaw
        )
        assert idx >= 0

    def test_target_idx_within_bounds(self):
        controller = StanleyController()
        cx, cy, cyaw = self._straight_path()
        _, idx = controller.compute_steering(
            x=25.0, y=0.0, yaw=0.0, v=2.0, cx=cx, cy=cy, cyaw=cyaw
        )
        assert 0 <= idx < len(cx)

    def test_last_target_idx_prevents_backtracking(self):
        controller = StanleyController(wheelbase=0.0)
        cx, cy, cyaw = self._straight_path(n=100)
        # Start near the end of the path
        _, idx = controller.compute_steering(
            x=40.0, y=0.0, yaw=0.0, v=2.0,
            cx=cx, cy=cy, cyaw=cyaw, last_target_idx=50,
        )
        assert idx >= 50

    # ------------------------------------------------------------------
    # Speed near zero
    # ------------------------------------------------------------------

    def test_low_speed_no_division_by_zero(self):
        controller = StanleyController(k=0.5, k_soft=1.0)
        cx, cy, cyaw = self._straight_path()
        # Should not raise even at v=0
        steer, _ = controller.compute_steering(
            x=0.0, y=0.5, yaw=0.0, v=0.0, cx=cx, cy=cy, cyaw=cyaw
        )
        assert math.isfinite(steer)

    # ------------------------------------------------------------------
    # Heading error component
    # ------------------------------------------------------------------

    def test_heading_error_correction(self):
        """Vehicle aligned with path but heading off by 10° → steering includes that correction."""
        controller = StanleyController(k=0.0, max_steer=math.radians(45), wheelbase=0.0)
        cx, cy, cyaw = self._straight_path()
        heading_offset = math.radians(10)
        steer, _ = controller.compute_steering(
            x=0.0, y=0.0, yaw=-heading_offset, v=5.0, cx=cx, cy=cy, cyaw=cyaw
        )
        # With k=0 only heading error contributes
        assert steer == pytest.approx(heading_offset, abs=1e-6)

    # ------------------------------------------------------------------
    # _normalise_angle
    # ------------------------------------------------------------------

    def test_normalise_angle_large_positive(self):
        result = StanleyController._normalise_angle(3 * math.pi)
        assert -math.pi < result <= math.pi

    def test_normalise_angle_large_negative(self):
        result = StanleyController._normalise_angle(-3 * math.pi)
        # -3π is the same as π; result should be in (-π, π]
        assert result == pytest.approx(math.pi, abs=1e-9)

    def test_normalise_angle_zero(self):
        assert StanleyController._normalise_angle(0.0) == pytest.approx(0.0)
