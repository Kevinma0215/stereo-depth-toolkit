from __future__ import annotations
import numpy as np
import cv2
from .presets import SGBMPreset

def create_matcher(p: SGBMPreset):
    mode = cv2.STEREO_SGBM_MODE_SGBM if p.mode == "SGBM" else cv2.STEREO_SGBM_MODE_HH
    matcher = cv2.StereoSGBM_create(
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
    return matcher

def compute_disparity(left_gray: np.ndarray, right_gray: np.ndarray, matcher) -> np.ndarray:
    disp = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disp

def disparity_to_depth(disp: np.ndarray, Q: np.ndarray) -> np.ndarray:
    # 使用 Q 做 reproject，取 Z
    points = cv2.reprojectImageTo3D(disp, Q)
    depth = points[:, :, 2].astype(np.float32)
    # 無效視差處理：disp<=0 或 depth 太大/太小視情況
    depth[disp <= 0.0] = np.nan
    return depth
