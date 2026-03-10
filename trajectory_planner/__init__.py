"""
TrajectoryPlanner - A Python library for trajectory planning and control.

Modules:
    poly_planner: Quintic polynomial trajectory planner for smooth path generation.
    stanley_controller: Stanley lateral controller for path tracking.
"""

from trajectory_planner.poly_planner import QuinticPolynomial, PolyPlanner
from trajectory_planner.stanley_controller import StanleyController

__all__ = ["QuinticPolynomial", "PolyPlanner", "StanleyController"]
__version__ = "0.1.0"
