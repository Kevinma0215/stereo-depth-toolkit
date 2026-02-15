""" 只負責「偵測 + 指標」，不碰檔案系統

這一步會解決你 got=0 難 debug 的問題：我們讓 detect 回傳「為何失敗」。 """
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class CharucoDetection:
    ok: bool
    num_markers: int
    num_charuco: int
    corners: np.ndarray | None
    ids: np.ndarray | None
    image_size: tuple[int, int]  # (w,h)
    reason: str | None = None

def detect_charuco(gray: np.ndarray, board, dictionary, *, min_markers: int = 4, min_charuco: int = 10) -> CharucoDetection:
    aruco = cv2.aruco
    h, w = gray.shape[:2]

    # 兼容新/舊 API
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary)

    if marker_ids is None:
        return CharucoDetection(False, 0, 0, None, None, (w, h), "no_markers")

    if len(marker_ids) < min_markers:
        return CharucoDetection(False, int(len(marker_ids)), 0, None, None, (w, h), "too_few_markers")

    # ret: (num_corners, charuco_corners, charuco_ids) (不同版本型態略有差異)
    ret = aruco.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=gray,
        board=board
    )
    num, ch_corners, ch_ids = ret

    if num is None or int(num) < min_charuco or ch_corners is None or ch_ids is None:
        return CharucoDetection(False, int(len(marker_ids)), int(num or 0), None, None, (w, h), "too_few_charuco")

    return CharucoDetection(True, int(len(marker_ids)), int(num), ch_corners, ch_ids, (w, h), None)
