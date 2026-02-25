from __future__ import annotations
import numpy as np
import cv2

from stereo_depth.use_cases.ports import IDepthEstimator
from stereo_depth.entities import CalibrationResult, DepthMap


class OpenCVDepthEstimator(IDepthEstimator):
    """IDepthEstimator backed by cv2.reprojectImageTo3D (Q-matrix method).

    ``left_rect`` in the returned DepthMap is left as None; StereoPipeline
    fills it with the actual rectified reference image.
    """

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
        depth = points[:, :, 2].astype(np.float32)
        depth[disparity <= 0.0] = np.nan
        return DepthMap(data=depth, disparity=disparity.copy())
