from __future__ import annotations
import numpy as np
import cv2

from stereo_depth.use_cases.ports import IDisparityMatcher
from stereo_depth.adapters.matcher.sgbm_presets import preset as _load_preset, SGBMPreset

_VALID_PRESETS = ("indoor", "outdoor", "high_quality")


class SgbmMatcher(IDisparityMatcher):
    """IDisparityMatcher backed by OpenCV StereoSGBM.

    Args:
        preset_name: One of 'indoor', 'outdoor', 'high_quality'.
                     Defaults to 'indoor'.
    """

    def __init__(self, preset_name: str = "indoor") -> None:
        if preset_name not in _VALID_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Valid options: {_VALID_PRESETS}"
            )
        p: SGBMPreset = _load_preset(preset_name)
        mode = (
            cv2.STEREO_SGBM_MODE_SGBM if p.mode == "SGBM"
            else cv2.STEREO_SGBM_MODE_HH
        )
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=p.min_disparity,
            numDisparities=p.num_disparities,
            blockSize=p.block_size,
            P1=p.p1,
            P2=p.p2,
            disp12MaxDiff=p.disp12_max_diff,
            preFilterCap=p.pre_filter_cap,
            uniquenessRatio=p.uniqueness_ratio,
            speckleWindowSize=p.speckle_window_size,
            speckleRange=p.speckle_range,
            mode=mode,
        )

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute disparity from a rectified stereo pair.

        Args:
            left:  uint8 image, shape (H, W) or (H, W, 3)
            right: uint8 image, same shape as left

        Returns:
            float32 disparity map, shape (H, W).  Invalid pixels have value <= 0.
        """
        left_gray = _to_gray(left)
        right_gray = _to_gray(right)
        disp = self._matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disp


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
