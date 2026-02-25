from __future__ import annotations
import numpy as np
import cv2

from stereo_depth.use_cases.ports import IRectifier
from stereo_depth.entities import FramePair, RectifiedPair, CalibrationResult


class OpenCVRectifier(IRectifier):
    """IRectifier backed by cv2.initUndistortRectifyMap + cv2.remap.

    Rectification maps are built lazily on the first call and cached by
    CalibrationResult object identity.  For a video pipeline always pass the
    same CalibrationResult instance to avoid rebuilding maps every frame.
    """

    def __init__(self) -> None:
        self._cache_id: int | None = None
        self._map1x: np.ndarray | None = None
        self._map1y: np.ndarray | None = None
        self._map2x: np.ndarray | None = None
        self._map2y: np.ndarray | None = None

    def _build_maps(self, calib: CalibrationResult) -> None:
        img_size = calib.image_size          # (width, height)
        D1 = calib.D1.reshape(-1, 1)
        D2 = calib.D2.reshape(-1, 1)
        self._map1x, self._map1y = cv2.initUndistortRectifyMap(
            calib.K1, D1, calib.R1, calib.P1, img_size, cv2.CV_32FC1
        )
        self._map2x, self._map2y = cv2.initUndistortRectifyMap(
            calib.K2, D2, calib.R2, calib.P2, img_size, cv2.CV_32FC1
        )
        self._cache_id = id(calib)

    def rectify(self, pair: FramePair, calib: CalibrationResult) -> RectifiedPair:
        if id(calib) != self._cache_id:
            self._build_maps(calib)
        left_r = cv2.remap(pair.left, self._map1x, self._map1y, cv2.INTER_LINEAR)
        right_r = cv2.remap(pair.right, self._map2x, self._map2y, cv2.INTER_LINEAR)
        return RectifiedPair(left=left_r, right=right_r)
