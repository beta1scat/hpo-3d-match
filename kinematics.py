"""Rotation and translation matrix utilities for 4x4 homogeneous transforms.

All rotation angles are in degrees.
"""

import numpy as np
from math import pi, cos, sin


def rotX(angle_deg: float) -> np.ndarray:
    """4x4 homogeneous rotation matrix about X axis."""
    a = angle_deg * pi / 180.0
    return np.array(
        [[1, 0, 0, 0], [0, cos(a), -sin(a), 0], [0, sin(a), cos(a), 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def rotY(angle_deg: float) -> np.ndarray:
    """4x4 homogeneous rotation matrix about Y axis."""
    a = angle_deg * pi / 180.0
    return np.array(
        [[cos(a), 0, sin(a), 0], [0, 1, 0, 0], [-sin(a), 0, cos(a), 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def rotZ(angle_deg: float) -> np.ndarray:
    """4x4 homogeneous rotation matrix about Z axis."""
    a = angle_deg * pi / 180.0
    return np.array(
        [[cos(a), -sin(a), 0, 0], [sin(a), cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def transXYZ(x: float, y: float, z: float) -> np.ndarray:
    """4x4 homogeneous translation matrix."""
    return np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float64
    )
