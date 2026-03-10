"""Unit tests for QuinticPolynomial and PolyPlanner."""

import math
import pytest

from trajectory_planner.poly_planner import QuinticPolynomial, PolyPlanner


class TestQuinticPolynomial:
    """Tests for QuinticPolynomial."""

    def _make_poly(self, xs=0.0, vs=0.0, as_=0.0, xe=10.0, ve=0.0, ae=0.0, T=10.0):
        return QuinticPolynomial(xs, vs, as_, xe, ve, ae, T)

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def test_start_position(self):
        poly = self._make_poly(xs=3.0)
        assert poly.position(0.0) == pytest.approx(3.0, abs=1e-9)

    def test_end_position(self):
        poly = self._make_poly(xs=0.0, xe=10.0, T=5.0)
        assert poly.position(5.0) == pytest.approx(10.0, abs=1e-6)

    def test_start_velocity(self):
        poly = self._make_poly(vs=2.0)
        assert poly.velocity(0.0) == pytest.approx(2.0, abs=1e-9)

    def test_end_velocity(self):
        poly = self._make_poly(ve=1.5, T=5.0)
        assert poly.velocity(5.0) == pytest.approx(1.5, abs=1e-6)

    def test_start_acceleration(self):
        poly = self._make_poly(as_=1.0)
        assert poly.acceleration(0.0) == pytest.approx(1.0, abs=1e-9)

    def test_end_acceleration(self):
        poly = self._make_poly(ae=0.5, T=5.0)
        assert poly.acceleration(5.0) == pytest.approx(0.5, abs=1e-6)

    # ------------------------------------------------------------------
    # Invalid duration
    # ------------------------------------------------------------------

    def test_zero_duration_raises(self):
        with pytest.raises(ValueError):
            QuinticPolynomial(0, 0, 0, 1, 0, 0, T=0.0)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError):
            QuinticPolynomial(0, 0, 0, 1, 0, 0, T=-1.0)

    # ------------------------------------------------------------------
    # Trivial case (start == end, all zeros)
    # ------------------------------------------------------------------

    def test_trivial_zero_motion(self):
        poly = QuinticPolynomial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, T=5.0)
        assert poly.position(2.5) == pytest.approx(0.0, abs=1e-9)
        assert poly.velocity(2.5) == pytest.approx(0.0, abs=1e-9)
        assert poly.acceleration(2.5) == pytest.approx(0.0, abs=1e-9)
        assert poly.jerk(2.5) == pytest.approx(0.0, abs=1e-9)

    # ------------------------------------------------------------------
    # Jerk is the derivative of acceleration
    # ------------------------------------------------------------------

    def test_jerk_is_derivative_of_acceleration(self):
        poly = self._make_poly(xs=0.0, xe=10.0, vs=1.0, ve=2.0, T=5.0)
        eps = 1e-5
        t = 2.5
        numerical_jerk = (poly.acceleration(t + eps) - poly.acceleration(t - eps)) / (
            2 * eps
        )
        assert poly.jerk(t) == pytest.approx(numerical_jerk, rel=1e-4)


class TestPolyPlanner:
    """Tests for PolyPlanner."""

    def _make_planner(self, **kwargs):
        defaults = dict(max_accel=10.0, max_jerk=10.0, dt=0.1, min_T=5.0, max_T=100.0, d_T=5.0)
        defaults.update(kwargs)
        return PolyPlanner(**defaults)

    # ------------------------------------------------------------------
    # Successful plan
    # ------------------------------------------------------------------

    def test_plan_returns_result(self):
        planner = self._make_planner()
        result = planner.plan(
            sx=0.0, sy=0.0, syaw=0.0, sv=0.0, sa=0.0,
            gx=10.0, gy=0.0, gyaw=0.0, gv=0.0, ga=0.0,
        )
        assert result is not None

    def test_plan_start_position(self):
        planner = self._make_planner()
        result = planner.plan(
            sx=1.0, sy=2.0, syaw=0.0, sv=0.0, sa=0.0,
            gx=10.0, gy=5.0, gyaw=0.0, gv=0.0, ga=0.0,
        )
        assert result is not None
        assert result.x[0] == pytest.approx(1.0, abs=1e-6)
        assert result.y[0] == pytest.approx(2.0, abs=1e-6)

    def test_plan_end_position(self):
        planner = self._make_planner()
        result = planner.plan(
            sx=0.0, sy=0.0, syaw=0.0, sv=0.0, sa=0.0,
            gx=10.0, gy=5.0, gyaw=0.0, gv=0.0, ga=0.0,
        )
        assert result is not None
        assert result.x[-1] == pytest.approx(10.0, abs=1e-4)
        assert result.y[-1] == pytest.approx(5.0, abs=1e-4)

    def test_plan_timestamps_increasing(self):
        planner = self._make_planner()
        result = planner.plan(
            sx=0.0, sy=0.0, syaw=0.0, sv=0.0, sa=0.0,
            gx=10.0, gy=0.0, gyaw=0.0, gv=0.0, ga=0.0,
        )
        assert result is not None
        for i in range(1, len(result.t)):
            assert result.t[i] > result.t[i - 1]

    def test_plan_lists_same_length(self):
        planner = self._make_planner()
        result = planner.plan(
            sx=0.0, sy=0.0, syaw=0.0, sv=0.0, sa=0.0,
            gx=10.0, gy=0.0, gyaw=0.0, gv=0.0, ga=0.0,
        )
        assert result is not None
        n = len(result.t)
        assert len(result.x) == n
        assert len(result.y) == n
        assert len(result.yaw) == n
        assert len(result.v) == n
        assert len(result.a) == n
        assert len(result.jerk) == n

    def test_plan_feasibility_limits_respected(self):
        planner = self._make_planner(max_accel=1.0, max_jerk=0.5)
        result = planner.plan(
            sx=0.0, sy=0.0, syaw=0.0, sv=0.0, sa=0.0,
            gx=10.0, gy=0.0, gyaw=0.0, gv=0.0, ga=0.0,
        )
        assert result is not None
        assert all(abs(a) <= 1.0 + 1e-6 for a in result.a)
        assert all(abs(j) <= 0.5 + 1e-6 for j in result.jerk)

    # ------------------------------------------------------------------
    # Infeasible plan
    # ------------------------------------------------------------------

    def test_plan_returns_none_when_infeasible(self):
        # Extremely tight limits that cannot be satisfied
        planner = PolyPlanner(
            max_accel=1e-6, max_jerk=1e-6, dt=0.1, min_T=5.0, max_T=10.0, d_T=5.0
        )
        result = planner.plan(
            sx=0.0, sy=0.0, syaw=0.0, sv=0.0, sa=0.0,
            gx=1000.0, gy=0.0, gyaw=0.0, gv=0.0, ga=0.0,
        )
        assert result is None
