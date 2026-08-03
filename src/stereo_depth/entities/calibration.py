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


@dataclass
class MonoIntrinsics:
    """Intrinsic calibration of a single camera.

    ``K``/``D`` pair with RAW (distorted) images. The distortion model
    determines the length and meaning of ``D``:
      - "pinhole":  (5,)  Brown-Conrady  (k1, k2, p1, p2, k3)
      - "rational": (8,)  Brown-Conrady  (k1, k2, p1, p2, k3, k4, k5, k6)
      - "fisheye":  (4,)  equidistant    (k1, k2, k3, k4)  — cv2.fisheye order
    """
    model: str                   # "pinhole" | "rational" | "fisheye"
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray                # 3×3 intrinsic matrix
    D: np.ndarray                # distortion coefficients, model-dependent length
    rpe_px: float                # full-set RMS reprojection error in pixels
    views_used: int              # number of views used in the final fit
