from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DepthMap:
    data: np.ndarray                     # float32, shape (H, W), metres; NaN for invalid
    disparity: np.ndarray                # float32, shape (H, W)
    left_rect: Optional[np.ndarray] = field(default=None)   # uint8 BGR (H, W, 3)
    right_rect: Optional[np.ndarray] = field(default=None)  # uint8 BGR (H, W, 3)


@dataclass
class PointCloud:
    """Placeholder for future 3-D point cloud output."""
