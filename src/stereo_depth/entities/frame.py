from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FramePair:
    left: np.ndarray   # uint8 BGR, shape (H, W, 3)
    right: np.ndarray  # uint8 BGR, shape (H, W, 3)


@dataclass
class RectifiedPair:
    left: np.ndarray   # uint8 BGR, shape (H, W, 3)
    right: np.ndarray  # uint8 BGR, shape (H, W, 3)
