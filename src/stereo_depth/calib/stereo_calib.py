""" 
純「校正計算」(不做 argparse、不讀檔)
把你原本 main 內的大段校正，搬到這裡。
 """
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import cv2

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
        pts = corners[idx, 0, :].astype(np.float32)  # Nx2
        return pts

    ptsL = pick(lc, li, common)
    ptsR = pick(rc, ri, common)
    obj = chess_corners_3d[common].astype(np.float32)  # Nx3
    return obj, ptsL, ptsR

def run_stereo_calibration(
    l_corners, l_ids,
    r_corners, r_ids,
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

    # paired views
    n = min(len(l_corners), len(r_corners))
    if n < min_views:
        raise RuntimeError(f"Not enough valid paired views. got={n}, need>={min_views}")

    l_corners, l_ids = l_corners[:n], l_ids[:n]
    r_corners, r_ids = r_corners[:n], r_ids[:n]

    flags = 0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    retL, K1, D1, *_ = aruco.calibrateCameraCharuco(
        charucoCorners=l_corners, charucoIds=l_ids, board=board, imageSize=img_size,
        cameraMatrix=None, distCoeffs=None, flags=flags, criteria=crit
    )
    retR, K2, D2, *_ = aruco.calibrateCameraCharuco(
        charucoCorners=r_corners, charucoIds=r_ids, board=board, imageSize=img_size,
        cameraMatrix=None, distCoeffs=None, flags=flags, criteria=crit
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
        raise RuntimeError(f"After ID matching, not enough stereo views: {len(objpoints)} (need>={min_views})")

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
