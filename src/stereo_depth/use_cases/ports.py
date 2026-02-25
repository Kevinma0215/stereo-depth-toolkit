from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from stereo_depth.entities import (
    FramePair,
    RectifiedPair,
    CalibrationResult,
    DepthMap,
)


class ICameraSource(ABC):
    @abstractmethod
    def grab(self) -> FramePair:
        """Capture and return the next stereo frame pair."""
        ...


class ICalibrationRepo(ABC):
    @abstractmethod
    def load(self, path: str) -> CalibrationResult:
        """Load a CalibrationResult from the given file path."""
        ...

    @abstractmethod
    def save(self, result: CalibrationResult, path: str) -> None:
        """Persist a CalibrationResult to the given file path."""
        ...


class IRectifier(ABC):
    @abstractmethod
    def rectify(self, pair: FramePair, calib: CalibrationResult) -> RectifiedPair:
        """Undistort and rectify a raw stereo frame pair."""
        ...


class IDisparityMatcher(ABC):
    @abstractmethod
    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute a disparity map from a rectified stereo pair.

        Args:
            left:  uint8 or float32 image, shape (H, W) or (H, W, 3)
            right: same shape as left

        Returns:
            float32 disparity map, shape (H, W)
        """
        ...


class IDepthEstimator(ABC):
    @abstractmethod
    def to_depth(self, disparity: np.ndarray, calib: CalibrationResult) -> DepthMap:
        """Convert a disparity map to a metric DepthMap using the Q matrix."""
        ...
