import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml

def collect_charuco(img_paths, board, dictionary):
    aruco = cv2.aruco
    all_corners, all_ids = [], []
    img_size = None
    valid = 0

    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img_size is None:
            h, w = img.shape[:2]
            img_size = (w, h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # corners, ids, _ = aruco.detectMarkers(gray, dictionary)
        # Detect markers (compatible with new OpenCV API)
        if hasattr(aruco, "ArucoDetector"):
            detector = aruco.ArucoDetector(dictionary)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)


        if ids is None or len(ids) < 4:
            continue

        # ok, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
        #     markerCorners=corners,
        #     markerIds=ids,
        #     image=gray,
        #     board=board
        # )
        ret = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )
        if ret[0] is None or ret[0] < 10:
            continue

        ok, ch_corners, ch_ids = ret

        # if ok is None or ok < 10:
        #     continue

        all_corners.append(ch_corners)
        all_ids.append(ch_ids)
        valid += 1

    return all_corners, all_ids, img_size, valid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/calib_frames", help="folder with left/ and right/")
    ap.add_argument("--out", default="data/calib.yaml")
    ap.add_argument("--squares-x", type=int, default=7)
    ap.add_argument("--squares-y", type=int, default=5)
    ap.add_argument("--square-length", type=float, default=0.04, help="meters")
    ap.add_argument("--marker-length", type=float, default=0.02, help="meters")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--min-views", type=int, default=15)
    args = ap.parse_args()

    aruco = cv2.aruco
    dict_id = getattr(aruco, args.dict)
    dictionary = aruco.getPredefinedDictionary(dict_id)
    board = aruco.CharucoBoard((args.squares_x, args.squares_y),
                               args.square_length, args.marker_length,
                               dictionary)

    data = Path(args.data)
    left_paths = sorted((data / "left").glob("*.png"))
    right_paths = sorted((data / "right").glob("*.png"))
    assert len(left_paths) == len(right_paths), "left/right count mismatch"

    l_corners, l_ids, img_size, l_valid = collect_charuco(left_paths, board, dictionary)
    r_corners, r_ids, _, r_valid = collect_charuco(right_paths, board, dictionary)

    n = min(len(l_corners), len(r_corners))
    if n < args.min_views:
        raise RuntimeError(f"Not enough valid views. got={n}, need>={args.min_views}. "
                           f"Try extracting more pairs / better board coverage.")

    # Use only first n views (they correspond by index)
    l_corners, l_ids = l_corners[:n], l_ids[:n]
    r_corners, r_ids = r_corners[:n], r_ids[:n]

    print(f"Using {n} valid paired views. Image size: {img_size}")

    # Calibrate each camera intrinsics
    flags = 0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    retL, K1, D1, rvecsL, tvecsL = aruco.calibrateCameraCharuco(
        charucoCorners=l_corners,
        charucoIds=l_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=crit
    )
    retR, K2, D2, rvecsR, tvecsR = aruco.calibrateCameraCharuco(
        charucoCorners=r_corners,
        charucoIds=r_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=crit
    )
    print("Mono reproj err L:", float(retL), "R:", float(retR))

    # Stereo calibration (fix intrinsics to stabilize)
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    # Need corresponding charuco corners per view:
    # stereoCalibrate expects corner points; we can feed the detected ChArUco corners directly
    # but must match IDs between L/R per view.
    objpoints = []
    imgpointsL = []
    imgpointsR = []

    # Board chessboard corners in 3D (board coordinate)
    # board.getChessboardCorners() returns Nx3 coordinates
    chess_corners_3d = board.getChessboardCorners()

    for lc, li, rc, ri in zip(l_corners, l_ids, r_corners, r_ids):
        li = li.flatten()
        ri = ri.flatten()

        common = np.intersect1d(li, ri)
        if common.size < 10:
            continue

        # Build per-view correspondences in same order of common ids
        def pick(corners, ids, common_ids):
            ids = ids.flatten()
            idx = [int(np.where(ids == cid)[0][0]) for cid in common_ids]
            pts = corners[idx, 0, :].astype(np.float32)  # Nx2
            return pts

        ptsL = pick(lc, li, common)
        ptsR = pick(rc, ri, common)
        obj = chess_corners_3d[common].astype(np.float32)  # Nx3

        objpoints.append(obj)
        imgpointsL.append(ptsL)
        imgpointsR.append(ptsR)

    if len(objpoints) < args.min_views:
        raise RuntimeError(f"After ID matching, not enough stereo views: {len(objpoints)}")

    rms, K1s, D1s, K2s, D2s, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpointsL,
        imagePoints2=imgpointsR,
        cameraMatrix1=K1,
        distCoeffs1=D1,
        cameraMatrix2=K2,
        distCoeffs2=D2,
        imageSize=img_size,
        criteria=crit,
        flags=stereo_flags
    )
    print("Stereo RMS:", float(rms))
    baseline = float(np.linalg.norm(T))
    print("Baseline (m):", baseline)

    # Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K1, distCoeffs1=D1,
        cameraMatrix2=K2, distCoeffs2=D2,
        imageSize=img_size,
        R=R, T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0  # crop to valid region
    )

    calib = {
        "image_size": {"width": img_size[0], "height": img_size[1]},
        "K1": K1.tolist(),
        "D1": D1.flatten().tolist(),
        "K2": K2.tolist(),
        "D2": D2.flatten().tolist(),
        "R": R.tolist(),
        "T": T.flatten().tolist(),
        "baseline_m": baseline,
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)
    print(f"Saved calibration to: {args.out}")

if __name__ == "__main__":
    main()
