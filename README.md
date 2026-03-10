# TrajectoryPlanner

A Python library for autonomous-vehicle trajectory planning and lateral control, featuring:

- **Quintic polynomial planner** – generates smooth, time-parameterised trajectories that satisfy configurable acceleration and jerk limits.
- **Stanley controller** – a classic lateral path-tracking controller that minimises heading error and cross-track error.

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Polynomial Trajectory Planner

```python
from trajectory_planner import PolyPlanner

planner = PolyPlanner(max_accel=1.0, max_jerk=0.5)
result = planner.plan(
    sx=0.0, sy=0.0, syaw=0.0, sv=0.0, sa=0.0,   # start state
    gx=10.0, gy=5.0, gyaw=0.0, gv=0.0, ga=0.0,  # goal state
)

if result is not None:
    print("x:", result.x)
    print("y:", result.y)
    print("time:", result.t)
```

### Stanley Controller

```python
import numpy as np
from trajectory_planner import StanleyController

# Reference path
cx = np.linspace(0, 50, 200)
cy = np.zeros(200)
cyaw = np.zeros(200)

controller = StanleyController(k=0.5, max_steer=np.radians(30))

steer, target_idx = controller.compute_steering(
    x=0.0, y=0.5, yaw=0.0, v=5.0,
    cx=cx, cy=cy, cyaw=cyaw,
)
print(f"Steering angle: {np.degrees(steer):.2f}°")
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Project Structure

```
trajectory_planner/
├── __init__.py             # Package entry-point
├── poly_planner.py         # QuinticPolynomial + PolyPlanner
└── stanley_controller.py   # StanleyController
tests/
├── test_poly_planner.py
└── test_stanley_controller.py
requirements.txt
setup.py
```

---

## License

MIT © YANG JIAXIN

