"""ChArUco stereo calibration — consolidated from calib/boards + detect + collect + stereo_calib + rectify.

This module is the canonical location for all ChArUco-related calibration logic.
The old ``stereo_depth.calib.*`` submodules are shims that re-export from here.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Board / dictionary construction  (from calib/boards.py)
# ---------------------------------------------------------------------------

def make_dictionary(name: str):
    aruco = cv2.aruco
    dict_id = getattr(aruco, name)
    return aruco.getPredefinedDictionary(dict_id)


def make_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dict_name: str = "DICT_4X4_50",
):
    aruco = cv2.aruco
    dictionary = make_dictionary(dict_name)
    board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    return board, dictionary


# ---------------------------------------------------------------------------
# Marker / corner detection  (from calib/detect.py)
# ---------------------------------------------------------------------------

@dataclass
class CharucoDetection:
    ok: bool
    num_markers: int
    num_charuco: int
    corners: np.ndarray | None
    ids: np.ndarray | None
    image_size: tuple[int, int]   # (w, h)
    reason: str | None = None


def detect_charuco(
    gray: np.ndarray,
    board,
    dictionary,
    *,
    min_markers: int = 4,
    min_charuco: int = 10,
) -> CharucoDetection:
    aruco = cv2.aruco
    h, w = gray.shape[:2]

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary)

    if marker_ids is None:
        return CharucoDetection(False, 0, 0, None, None, (w, h), "no_markers")

    if len(marker_ids) < min_markers:
        return CharucoDetection(False, int(len(marker_ids)), 0, None, None, (w, h), "too_few_markers")

    ret = aruco.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=gray,
        board=board,
    )
    num, ch_corners, ch_ids = ret

    if num is None or int(num) < min_charuco or ch_corners is None or ch_ids is None:
        return CharucoDetection(
            False, int(len(marker_ids)), int(num or 0), None, None, (w, h), "too_few_charuco"
        )

    return CharucoDetection(True, int(len(marker_ids)), int(num), ch_corners, ch_ids, (w, h), None)


# ---------------------------------------------------------------------------
# Image collection + reporting  (from calib/collect.py)
# ---------------------------------------------------------------------------

@dataclass
class CollectReport:
    total: int
    ok: int
    fail_no_markers: int
    fail_too_few_markers: int
    fail_too_few_charuco: int
    min_charuco_seen: int
    max_charuco_seen: int


def collect_charuco_from_paths(
    img_paths: list[Path],
    board,
    dictionary,
    *,
    min_markers: int = 4,
    min_charuco: int = 10,
):
    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None

    stats = {
        "total": 0, "ok": 0,
        "no_markers": 0, "too_few_markers": 0, "too_few_charuco": 0,
        "min_charuco": 10**9, "max_charuco": 0,
    }

    for p in img_paths:
        stats["total"] += 1
        img = cv2.imread(str(p))
        if img is None:
            stats["no_markers"] += 1
            continue

        if img_size is None:
            h, w = img.shape[:2]
            img_size = (w, h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        det: CharucoDetection = detect_charuco(
            gray, board, dictionary, min_markers=min_markers, min_charuco=min_charuco
        )

        stats["min_charuco"] = min(stats["min_charuco"], det.num_charuco)
        stats["max_charuco"] = max(stats["max_charuco"], det.num_charuco)

        if not det.ok:
            stats[det.reason or "no_markers"] += 1
            continue

        all_corners.append(det.corners)   # type: ignore[arg-type]
        all_ids.append(det.ids)            # type: ignore[arg-type]
        stats["ok"] += 1

    if img_size is None:
        img_size = (0, 0)

    report = CollectReport(
        total=stats["total"],
        ok=stats["ok"],
        fail_no_markers=stats["no_markers"],
        fail_too_few_markers=stats["too_few_markers"],
        fail_too_few_charuco=stats["too_few_charuco"],
        min_charuco_seen=(0 if stats["min_charuco"] == 10**9 else stats["min_charuco"]),
        max_charuco_seen=stats["max_charuco"],
    )
    return all_corners, all_ids, img_size, report


# ---------------------------------------------------------------------------
# Stereo calibration math  (from calib/stereo_calib.py)
# ---------------------------------------------------------------------------

@dataclass
class StereoCalibResult:
    image_size: tuple[int, int]
    K1: list[list[float]]
    D1: list[float]
    K2: list[list[float]]
    D2: list[float]
    R: list[list[float]]
    T: list[float]
    baseline_m: float
    R1: list[list[float]]
    R2: list[list[float]]
    P1: list[list[float]]
    P2: list[list[float]]
    Q: list[list[float]]
    mono_reproj_L: float
    mono_reproj_R: float
    stereo_rms: float
    used_views: int
    matched_views: int


def _match_ids_one_view(lc, li, rc, ri, chess_corners_3d, min_common: int = 10):
    li = li.flatten()
    ri = ri.flatten()
    common = np.intersect1d(li, ri)
    if common.size < min_common:
        return None

    def pick(corners, ids, common_ids):
        ids = ids.flatten()
        idx = [int(np.where(ids == cid)[0][0]) for cid in common_ids]
        pts = corners[idx, 0, :].astype(np.float32)
        return pts

    ptsL = pick(lc, li, common)
    ptsR = pick(rc, ri, common)
    obj = chess_corners_3d[common].astype(np.float32)
    return obj, ptsL, ptsR


def run_stereo_calibration(
    l_corners,
    l_ids,
    r_corners,
    r_ids,
    img_size: tuple[int, int],
    board,
    *,
    min_views: int = 15,
    min_common_ids: int = 10,
):
    aruco = cv2.aruco
    w, h = img_size
    if w <= 0 or h <= 0:
        raise RuntimeError("Invalid image_size")

    n = min(len(l_corners), len(r_corners))
    if n < min_views:
        raise RuntimeError(f"Not enough valid paired views. got={n}, need>={min_views}")

    l_corners, l_ids = l_corners[:n], l_ids[:n]
    r_corners, r_ids = r_corners[:n], r_ids[:n]

    flags = 0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    retL, K1, D1, *_ = aruco.calibrateCameraCharuco(
        charucoCorners=l_corners, charucoIds=l_ids, board=board, imageSize=img_size,
        cameraMatrix=None, distCoeffs=None, flags=flags, criteria=crit,
    )
    retR, K2, D2, *_ = aruco.calibrateCameraCharuco(
        charucoCorners=r_corners, charucoIds=r_ids, board=board, imageSize=img_size,
        cameraMatrix=None, distCoeffs=None, flags=flags, criteria=crit,
    )

    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    chess_corners_3d = board.getChessboardCorners()

    objpoints, imgpointsL, imgpointsR = [], [], []
    for lc, li, rc, ri in zip(l_corners, l_ids, r_corners, r_ids):
        matched = _match_ids_one_view(lc, li, rc, ri, chess_corners_3d, min_common=min_common_ids)
        if matched is None:
            continue
        obj, ptsL, ptsR = matched
        objpoints.append(obj)
        imgpointsL.append(ptsL)
        imgpointsR.append(ptsR)

    if len(objpoints) < min_views:
        raise RuntimeError(
            f"After ID matching, not enough stereo views: {len(objpoints)} (need>={min_views})"
        )

    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpointsL,
        imagePoints2=imgpointsR,
        cameraMatrix1=K1, distCoeffs1=D1,
        cameraMatrix2=K2, distCoeffs2=D2,
        imageSize=img_size,
        criteria=crit,
        flags=stereo_flags,
    )

    baseline = float(np.linalg.norm(T))

    R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(
        cameraMatrix1=K1, distCoeffs1=D1,
        cameraMatrix2=K2, distCoeffs2=D2,
        imageSize=img_size,
        R=R, T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    return StereoCalibResult(
        image_size=img_size,
        K1=K1.tolist(), D1=D1.flatten().tolist(),
        K2=K2.tolist(), D2=D2.flatten().tolist(),
        R=R.tolist(), T=T.flatten().tolist(),
        baseline_m=baseline,
        R1=R1.tolist(), R2=R2.tolist(),
        P1=P1.tolist(), P2=P2.tolist(),
        Q=Q.tolist(),
        mono_reproj_L=float(retL),
        mono_reproj_R=float(retR),
        stereo_rms=float(rms),
        used_views=n,
        matched_views=len(objpoints),
    )


# ---------------------------------------------------------------------------
# Rectification utilities  (from calib/rectify.py)
# ---------------------------------------------------------------------------

@dataclass
class RectifyMaps:
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    image_size: tuple[int, int]   # (w, h)


def build_rectify_maps(calib: dict) -> RectifyMaps:
    w = int(calib["image_size"]["width"])
    h = int(calib["image_size"]["height"])
    img_size = (w, h)

    K1 = np.array(calib["K1"], dtype=np.float64)
    D1 = np.array(calib["D1"], dtype=np.float64).reshape(-1, 1)
    K2 = np.array(calib["K2"], dtype=np.float64)
    D2 = np.array(calib["D2"], dtype=np.float64).reshape(-1, 1)
    R1 = np.array(calib["R1"], dtype=np.float64)
    R2 = np.array(calib["R2"], dtype=np.float64)
    P1 = np.array(calib["P1"], dtype=np.float64)
    P2 = np.array(calib["P2"], dtype=np.float64)

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    return RectifyMaps(
        map1x=map1x, map1y=map1y,
        map2x=map2x, map2y=map2y,
        image_size=img_size,
    )


def rectify_pair(left_bgr: np.ndarray, right_bgr: np.ndarray, maps: RectifyMaps):
    left_r = cv2.remap(left_bgr, maps.map1x, maps.map1y, interpolation=cv2.INTER_LINEAR)
    right_r = cv2.remap(right_bgr, maps.map2x, maps.map2y, interpolation=cv2.INTER_LINEAR)
    return left_r, right_r
