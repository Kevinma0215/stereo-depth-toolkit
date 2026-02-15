from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

@dataclass
class RectifyMaps:
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    image_size: tuple[int, int]  # (w, h)

def build_rectify_maps(calib: dict) -> RectifyMaps:
    w = int(calib["image_size"]["width"])
    h = int(calib["image_size"]["height"])
    img_size = (w, h)

    K1 = np.array(calib["K1"], dtype=np.float64)
    D1 = np.array(calib["D1"], dtype=np.float64).reshape(-1, 1)
    K2 = np.array(calib["K2"], dtype=np.float64)
    D2 = np.array(calib["D2"], dtype=np.float64).reshape(-1, 1)

    # 你存的 R1/R2/P1/P2
    R1 = np.array(calib["R1"], dtype=np.float64)
    R2 = np.array(calib["R2"], dtype=np.float64)
    P1 = np.array(calib["P1"], dtype=np.float64)
    P2 = np.array(calib["P2"], dtype=np.float64)

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    return RectifyMaps(map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y, image_size=img_size)

def rectify_pair(left_bgr: np.ndarray, right_bgr: np.ndarray, maps: RectifyMaps):
    left_r = cv2.remap(left_bgr, maps.map1x, maps.map1y, interpolation=cv2.INTER_LINEAR)
    right_r = cv2.remap(right_bgr, maps.map2x, maps.map2y, interpolation=cv2.INTER_LINEAR)
    return left_r, right_r
