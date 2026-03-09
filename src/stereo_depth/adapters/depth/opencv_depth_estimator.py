from __future__ import annotations
import numpy as np
import cv2

from stereo_depth.use_cases.ports import IDepthEstimator
from stereo_depth.entities import CalibrationResult, DepthMap
from stereo_depth.adapters.post_processor.hole_fill_post_processor import (
    HoleFillPostProcessor,
)


class OpenCVDepthEstimator(IDepthEstimator):
    """IDepthEstimator backed by cv2.reprojectImageTo3D (Q-matrix method).

    ``left_rect`` in the returned DepthMap is left as None; StereoPipeline
    fills it with the actual rectified reference image.

    Args:
        fill_holes:  If True, run cv2.inpaint (INPAINT_NS) to fill NaN/zero
                     depth pixels after disparity→depth conversion.
                     Prefer passing a :class:`HoleFillPostProcessor` to
                     ``StereoPipeline`` instead; this flag is kept for
                     backward compatibility.
        fill_radius: Inpainting neighbourhood radius in pixels (default 3).
    """

    def __init__(self, fill_holes: bool = False, fill_radius: int = 3) -> None:
        if fill_radius < 1:
            raise ValueError(f"fill_radius must be >= 1, got {fill_radius}")
        self._fill_holes  = fill_holes
        self._fill_radius = fill_radius
        self._hole_filler = HoleFillPostProcessor(radius=fill_radius) if fill_holes else None

    def to_depth(self, disparity: np.ndarray, calib: CalibrationResult) -> DepthMap:
        """Convert a float32 disparity map to metric depth via the Q matrix.

        Args:
            disparity: float32 array, shape (H, W).  Pixels with value <= 0
                       are treated as invalid.
            calib:     CalibrationResult carrying the 4×4 Q matrix.

        Returns:
            DepthMap with ``data`` in metres (NaN for invalid pixels),
            ``disparity`` as-is, and ``left_rect=None``.
        """
        points = cv2.reprojectImageTo3D(disparity, calib.Q)
        depth  = points[:, :, 2].astype(np.float32)
        depth[disparity <= 0.0] = np.nan

        result = DepthMap(data=depth, disparity=disparity.copy())

        if self._hole_filler is not None:
            result = self._hole_filler.process(result)

        return result


# Convenience alias matching the example scripts
OpencvDepthEstimator = OpenCVDepthEstimator
