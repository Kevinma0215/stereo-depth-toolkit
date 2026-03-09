#!/usr/bin/env python3
"""ChArUco-based stereo depth accuracy evaluation.

Usage:
    python tools/eval_depth_accuracy.py \
        --calib outputs/calib/0308_try4/calib.yaml \
        --preset indoor --frames 10 --out outputs/accuracy_eval/

Workflow:
    1. Hold a ChArUco board in front of the camera.
    2. When "DETECTED" appears, press SPACE to capture a position.
       The tool accumulates --frames disparity frames into a fresh
       DepthAggregator, then queries every detected corner.
    3. Move to a different distance and repeat.
    4. Press Q (or ESC) to finish and generate the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from stereo_depth.adapters.calibration.charuco_calibrator import (
    detect_charuco,
    make_charuco_board,
)
from stereo_depth.adapters.calibration.yaml_repo import YamlCalibrationRepo
from stereo_depth.adapters.camera.uvc_source import UVCSource, open_source
from stereo_depth.adapters.depth.opencv_depth_estimator import OpenCVDepthEstimator
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.infrastructure.depth_aggregator import DepthAggregator
from stereo_depth.infrastructure.io.sbs_capture import SBSSplitter
from stereo_depth.use_cases.pipeline import StereoPipeline

_WIN  = "Depth Accuracy Eval"
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Threshold / verdict helpers
# ---------------------------------------------------------------------------

def _threshold_mm(gt_z_m: float) -> float:
    """PASS threshold (mm) by depth zone."""
    if gt_z_m < 0.65:
        return 15.0
    if gt_z_m <= 1.0:
        return 35.0
    return 70.0


def _verdict(mean_err_mm: float, gt_z_m: float) -> str:
    thr = _threshold_mm(gt_z_m)
    if mean_err_mm < thr:
        return "PASS"
    if mean_err_mm < 2.0 * thr:
        return "MARGINAL"
    return "FAIL"


def _verdict_symbol(v: str) -> str:
    return {"PASS": "✓", "MARGINAL": "~", "FAIL": "✗"}.get(v, "?")


# ---------------------------------------------------------------------------
# solvePnP on the rectified left image
# ---------------------------------------------------------------------------

def _solve_pnp(
    det,
    board,
    K_rect: np.ndarray,
):
    """Run solvePnP using detected ChArUco corners and the rectified K matrix.

    The image is already undistorted/rectified, so distortion is zero.
    K_rect = P1[:3, :3] from calib.

    Returns (rvec, tvec, obj_pts_3d, img_pts_2d) or None on failure.
    """
    if det.ids is None or det.corners is None or len(det.ids) < 4:
        return None

    ids_flat    = det.ids.flatten()
    all_obj_pts = board.getChessboardCorners()          # (total, 3)
    obj_pts     = all_obj_pts[ids_flat].astype(np.float32)   # (N, 3)
    img_pts     = det.corners.reshape(-1, 2).astype(np.float32)  # (N, 2)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts,
        K_rect.astype(np.float32),
        np.zeros(5, dtype=np.float32),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec, tvec, obj_pts, img_pts


def _gt_xyz_camera_frame(rvec, tvec, obj_pts: np.ndarray) -> np.ndarray:
    """Transform board-frame 3-D points into camera frame.  Returns (N, 3)."""
    R, _ = cv2.Rodrigues(rvec)
    return (R @ obj_pts.T + tvec).T   # (N, 3)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _draw_progress(canvas: np.ndarray, done: int, total: int) -> None:
    bar_w  = 20
    filled = int(bar_w * done / max(total, 1))
    bar    = "=" * filled + " " * (bar_w - filled)
    cv2.putText(
        canvas, f"Accumulating... [{bar}] {done}/{total}",
        (10, 30), _FONT, 0.70, (0, 255, 255), 2, cv2.LINE_AA,
    )


def _draw_detection_overlay(
    canvas: np.ndarray, det, pnp_result, board,
) -> None:
    cv2.aruco.drawDetectedCornersCharuco(canvas, det.corners, det.ids, (0, 255, 0))
    if pnp_result is not None:
        _, tvec, _, _ = pnp_result
        z = float(tvec[2][0])
        cv2.putText(
            canvas,
            f"DETECTED \u2014 solvePnP Z: {z:.3f} m | SPACE=capture  Q=finish",
            (8, 28), _FONT, 0.60, (0, 255, 0), 2, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas, "DETECTED (solvePnP failed \u2014 reposition)",
            (8, 28), _FONT, 0.60, (0, 165, 255), 2, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Per-position statistics
# ---------------------------------------------------------------------------

def _compute_stats(position: dict) -> dict:
    errors = [c["error_mm"] for c in position["corners"] if c["error_mm"] is not None]
    confs  = [c["confidence"] for c in position["corners"] if c["error_mm"] is not None]
    skipped = sum(1 for c in position["corners"] if c["error_mm"] is None)
    gt_z    = position["gt_z_m"]

    if not errors:
        return dict(
            gt_z_m=gt_z, verdict="FAIL",
            mean_err_mm=None, median_err_mm=None, p95_err_mm=None,
            mean_confidence=None, evaluated=0, skipped=skipped,
        )

    mean_e = float(np.mean(errors))
    return dict(
        gt_z_m=gt_z,
        verdict=_verdict(mean_e, gt_z),
        mean_err_mm=mean_e,
        median_err_mm=float(np.median(errors)),
        p95_err_mm=float(np.percentile(errors, 95)),
        mean_confidence=float(np.mean(confs)),
        evaluated=len(errors),
        skipped=skipped,
    )


def _print_position_summary(pos_idx: int, stats: dict) -> None:
    gt_z = stats["gt_z_m"]
    print(f"\nPosition {pos_idx}  (gt Z: {gt_z:.3f} m)")
    print("\u2500" * 48)
    print(f"  corners evaluated  : {stats['evaluated']}")
    print(f"  corners skipped    : {stats['skipped']}  (low confidence)")
    if stats["mean_err_mm"] is not None:
        sym = _verdict_symbol(stats["verdict"])
        print(f"  mean error         : {stats['mean_err_mm']:.1f} mm  {sym} {stats['verdict']}")
        print(f"  median error       : {stats['median_err_mm']:.1f} mm")
        print(f"  p95 error          : {stats['p95_err_mm']:.1f} mm")
        print(f"  mean confidence    : {stats['mean_confidence'] * 100:.0f}%")
    else:
        print("  (no valid corners — position excluded from report)")


# ---------------------------------------------------------------------------
# Capture sequence (called on SPACE key)
# ---------------------------------------------------------------------------

def _run_capture_sequence(
    source,
    pipeline,
    calib,
    frozen_det,
    frozen_pnp,
    n_frames: int,
) -> dict | None:
    rvec, tvec, obj_pts, img_pts = frozen_pnp
    gt_xyz  = _gt_xyz_camera_frame(rvec, tvec, obj_pts)    # (N, 3)
    gt_z_m  = float(np.median(gt_xyz[:, 2]))
    pixels  = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in img_pts]

    aggregator = DepthAggregator(
        calib=calib,
        buffer_size=n_frames,
        min_confidence=0.4,
        roi_radius=4,
    )

    for done in range(1, n_frames + 1):
        pair      = source.grab()
        depth_map = pipeline.process(pair)
        aggregator.push(depth_map.disparity)

        canvas = (
            depth_map.left_rect.copy()
            if depth_map.left_rect is not None
            else pair.left.copy()
        )
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        _draw_progress(canvas, done, n_frames)
        cv2.aruco.drawDetectedCornersCharuco(
            canvas, frozen_det.corners, frozen_det.ids, (0, 200, 0),
        )
        cv2.imshow(_WIN, canvas)
        cv2.waitKey(1)

    wp_results = aggregator.query(pixels)

    corners_out = []
    for i, (wp, gt) in enumerate(zip(wp_results, gt_xyz)):
        u, v = pixels[i]
        if wp.xyz_m is not None and wp.confidence >= 0.4:
            stereo = np.array(wp.xyz_m, dtype=np.float64)
            err_mm = float(np.linalg.norm(gt - stereo) * 1000.0)
        else:
            stereo = None
            err_mm = None

        corners_out.append({
            "pixel":        [u, v],
            "gt_xyz_m":     [float(gt[0]), float(gt[1]), float(gt[2])],
            "stereo_xyz_m": [float(x) for x in wp.xyz_m] if wp.xyz_m else None,
            "error_mm":     err_mm,
            "confidence":   float(wp.confidence),
        })

    return {"gt_z_m": gt_z_m, "verdict": "", "corners": corners_out}


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def _overall_verdict(stats_list: list[dict]) -> str:
    vs = [s["verdict"] for s in stats_list]
    if "FAIL"     in vs: return "FAIL"
    if "MARGINAL" in vs: return "MARGINAL"
    return "PASS"


def _usable_range(stats_list: list[dict]) -> float | None:
    last = None
    for s in stats_list:
        if s["verdict"] == "PASS":
            last = s["gt_z_m"]
    return last


def _mm(v: float | None) -> str:
    return f"{v:6.1f} mm" if v is not None else "     N/A"


def _pct(v: float | None) -> str:
    return f"{v * 100:5.0f}%" if v is not None else "  N/A"


def _build_report(
    calib_path: str,
    preset: str,
    frames: int,
    stats_list: list[dict],
) -> str:
    now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall = _overall_verdict(stats_list)
    usable  = _usable_range(stats_list)
    sym_ov  = _verdict_symbol(overall)

    H = "  "
    lines: list[str] = []
    lines += [
        "",
        H + "Stereo Depth Accuracy Report",
        H + "\u2550" * 50,
        H + f"calib    : {calib_path}",
        H + f"preset   : {preset}",
        H + f"frames   : {frames}",
        H + f"date     : {now}",
        H + "primary working distance: ~0.55 m",
        "",
        H + "Per-position results:",
        H + "\u250c" + "\u2500" * 6 + "\u252c" + "\u2500" * 9 + "\u252c" + "\u2500" * 11
          + "\u252c" + "\u2500" * 12 + "\u252c" + "\u2500" * 10
          + "\u252c" + "\u2500" * 10 + "\u252c" + "\u2500" * 9 + "\u2510",
        H + "\u2502  Pos \u2502  GT Z   \u2502 Mean err  \u2502 Median err "
          + "\u2502 P95 err  \u2502 Conf avg \u2502 Verdict \u2502",
        H + "\u251c" + "\u2500" * 6 + "\u253c" + "\u2500" * 9 + "\u253c" + "\u2500" * 11
          + "\u253c" + "\u2500" * 12 + "\u253c" + "\u2500" * 10
          + "\u253c" + "\u2500" * 10 + "\u253c" + "\u2500" * 9 + "\u2524",
    ]

    for i, s in enumerate(stats_list, start=1):
        vsym = _verdict_symbol(s["verdict"])
        row  = (
            H
            + f"\u2502 {i:3d}  "
            + f"\u2502 {s['gt_z_m']:.2f} m  "
            + f"\u2502 {_mm(s['mean_err_mm']):>9s} "
            + f"\u2502 {_mm(s['median_err_mm']):>10s} "
            + f"\u2502 {_mm(s['p95_err_mm']):>8s} "
            + f"\u2502 {_pct(s['mean_confidence']):>8s} "
            + f"\u2502 {s['verdict']} {vsym:<5s}\u2502"
        )
        lines.append(row)

    lines += [
        H + "\u2514" + "\u2500" * 6 + "\u2534" + "\u2500" * 9 + "\u2534" + "\u2500" * 11
          + "\u2534" + "\u2500" * 12 + "\u2534" + "\u2500" * 10
          + "\u2534" + "\u2500" * 10 + "\u2534" + "\u2500" * 9 + "\u2518",
        "",
        H + f"Overall verdict: {overall} {sym_ov}",
    ]

    if usable is not None:
        lines.append(H + f"Usable range   : {usable:.2f} m  (last PASS position)")
    else:
        lines.append(H + "Usable range   : none")

    lines += [
        "",
        H + "\u2500\u2500 Recommendation for VLA pick-and-place at 55cm "
          + "\u2500" * 26,
    ]

    if overall == "PASS":
        lines.append(
            H + '"Depth accuracy sufficient for pick-and-place. Proceed to\n'
            + H + ' workspace homography calibration."'
        )
    elif overall == "MARGINAL":
        lines.append(
            H + '"Accuracy borderline. Consider reducing WLS sigma or\n'
            + H + ' increasing buffer frames to 15 before proceeding."'
        )
    else:
        lines.append(
            H + '"Accuracy insufficient. Re-run stereo calibration and\n'
            + H + ' re-evaluate before building on this depth source."'
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ChArUco-based stereo depth accuracy evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--calib",         required=True,
                   help="Path to calib.yaml")
    p.add_argument("--preset",        default="indoor",
                   help="SGBM preset name")
    p.add_argument("--frames",        type=int, default=10,
                   help="Disparity frames to accumulate per capture")
    p.add_argument("--out",           default="outputs/accuracy_eval/",
                   help="Output directory for report and JSON")
    p.add_argument("--dict-name",     default="DICT_5X5_100",
                   help="ArUco dictionary name")
    p.add_argument("--square-length", type=float, default=0.03,
                   help="Board square side length (metres)")
    p.add_argument("--marker-length", type=float, default=0.022,
                   help="Board marker side length (metres)")
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load calibration ──────────────────────────────────────────────────
    calib  = YamlCalibrationRepo().load(str(args.calib))
    K_rect = np.array(calib.P1, dtype=np.float64)[:3, :3]

    # ── ChArUco board ─────────────────────────────────────────────────────
    board, dictionary = make_charuco_board(
        squares_x=7, squares_y=5,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dict_name=args.dict_name,
    )

    # ── Stereo pipeline ───────────────────────────────────────────────────
    pipeline = StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=SgbmMatcher(preset_name=args.preset),
        depth_estimator=OpenCVDepthEstimator(),
        calib=calib,
    )

    # ── Camera ────────────────────────────────────────────────────────────
    cap    = open_source(device=0, width=2560, height=720)
    source = UVCSource(cap, SBSSplitter())

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

    positions: list[dict] = []
    pos_num = 0

    print("=== Stereo Depth Accuracy Evaluation ===")
    print(f"  calib  : {args.calib}")
    print(f"  preset : {args.preset}")
    print(f"  frames : {args.frames}")
    print()
    print("  Hold ChArUco board in view.  SPACE = capture position.  Q = finish.")
    print()

    try:
        while True:
            pair      = source.grab()
            depth_map = pipeline.process(pair)

            # Build display canvas from rectified left image
            canvas = (
                depth_map.left_rect.copy()
                if depth_map.left_rect is not None
                else pair.left.copy()
            )
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

            # ChArUco detection on rectified image
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            det  = detect_charuco(gray, board, dictionary, min_markers=4, min_charuco=6)

            pnp_result = None
            if det.ok:
                pnp_result = _solve_pnp(det, board, K_rect)
                _draw_detection_overlay(canvas, det, pnp_result, board)
            else:
                cv2.putText(
                    canvas, "Searching for board...",
                    (8, 28), _FONT, 0.65, (0, 0, 255), 2, cv2.LINE_AA,
                )

            cv2.putText(
                canvas,
                f"Positions captured: {pos_num}  |  SPACE=capture  Q=finish",
                (8, canvas.shape[0] - 10), _FONT, 0.50, (200, 200, 200), 1, cv2.LINE_AA,
            )

            cv2.imshow(_WIN, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key == ord(" "):
                if not det.ok:
                    print("  [SKIP] No board detected — move board into view first.")
                    continue
                if pnp_result is None:
                    print("  [SKIP] solvePnP failed — reposition board and try again.")
                    continue

                pos_num += 1
                _, tvec, _, _ = pnp_result
                print(
                    f"  Capturing position {pos_num}  "
                    f"(approx Z: {float(tvec[2][0]):.3f} m) ..."
                )

                result = _run_capture_sequence(
                    source, pipeline, calib,
                    det, pnp_result, args.frames,
                )
                if result is None:
                    print(f"  [FAIL] Capture sequence returned no data — skipping.")
                    pos_num -= 1
                    continue

                stats        = _compute_stats(result)
                result["verdict"] = stats["verdict"]
                positions.append(result)
                _print_position_summary(pos_num, stats)

    finally:
        source.release()
        cv2.destroyAllWindows()

    if not positions:
        print("\nNo positions captured. Exiting without report.")
        return

    # ── Final report ──────────────────────────────────────────────────────
    stats_list  = [_compute_stats(p) for p in positions]
    report_text = _build_report(args.calib, args.preset, args.frames, stats_list)
    print(report_text)

    report_path = out_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Saved : {report_path}")

    raw = {
        "meta": {
            "calib":  args.calib,
            "preset": args.preset,
            "frames": args.frames,
            "date":   datetime.now().isoformat(),
        },
        "positions": positions,
    }
    errors_path = out_dir / "raw_errors.json"
    errors_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"  Saved : {errors_path}")


if __name__ == "__main__":
    main()
