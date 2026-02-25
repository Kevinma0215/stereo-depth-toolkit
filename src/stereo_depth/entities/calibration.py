from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CalibrationResult:
    image_size: tuple[int, int]  # (width, height)
    K1: np.ndarray               # 3×3 left intrinsic matrix
    D1: np.ndarray               # distortion coefficients (5,)
    K2: np.ndarray               # 3×3 right intrinsic matrix
    D2: np.ndarray               # distortion coefficients (5,)
    R: np.ndarray                # 3×3 rotation — right camera w.r.t. left
    T: np.ndarray                # translation vector (3,) in metres
    baseline_m: float
    R1: np.ndarray               # 3×3 rectification rotation for left
    R2: np.ndarray               # 3×3 rectification rotation for right
    P1: np.ndarray               # 3×4 projection matrix for left
    P2: np.ndarray               # 3×4 projection matrix for right
    Q: np.ndarray                # 4×4 disparity-to-depth reprojection matrix
    rpe_px: float                # reprojection error in pixels (informational)
