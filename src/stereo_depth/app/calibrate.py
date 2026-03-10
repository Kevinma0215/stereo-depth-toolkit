"""把「收集→校正→輸出→report」串起來（report 永遠產生）"""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

from stereo_depth.adapters.calibration.charuco_calibrator import (
    make_charuco_board, collect_charuco_paired, run_stereo_calibration,
    compute_per_image_rpe,
)
from stereo_depth.infrastructure.config.io import save_yaml


def _list_images(folder: Path) -> list[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    paths: list[Path] = []
    for e in exts:
        paths.extend(folder.glob(e))
    return sorted(paths)


def run_calibrate_charuco_stereo(
    data_dir: Path,
    out_yaml: Path,
    *,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length: float = 0.03,
    marker_length: float = 0.022,
    dict_name: str = "DICT_5X5_100",
    min_views: int = 15,
    min_markers: int = 4,
    min_charuco: int = 10,
    min_common_ids: int = 10,
    report_json: Path | None = None,
    per_image_rpe: bool = False,
):
    board, dictionary = make_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length,
        marker_length=marker_length,
        dict_name=dict_name,
    )

    left_dir = data_dir / "left"
    right_dir = data_dir / "right"

    left_paths = _list_images(left_dir)
    right_paths = _list_images(right_dir)

    if len(left_paths) == 0 or len(right_paths) == 0:
        # report 也要寫出來
        if report_json is None:
            report_json = out_yaml.with_suffix(".report.json")
        report = {
            "status": "failed",
            "reason": "no_images_found",
            "inputs": {
                "data_dir": str(data_dir),
                "left_dir": str(left_dir),
                "right_dir": str(right_dir),
                "num_left_images": len(left_paths),
                "num_right_images": len(right_paths),
            },
            "params": {
                "squares_x": squares_x,
                "squares_y": squares_y,
                "square_length": square_length,
                "marker_length": marker_length,
                "dict_name": dict_name,
                "min_views": min_views,
                "min_markers": min_markers,
                "min_charuco": min_charuco,
                "min_common_ids": min_common_ids,
            },
        }
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(f"No images found. See report: {report_json}")

    # Detect corners from paired images: only keep a view when BOTH sides succeed,
    # guaranteeing l_corners[i] and r_corners[i] always correspond to the same
    # physical board position.
    l_corners, l_ids, r_corners, r_ids, img_size, l_report, r_report, matched_left_names = (
        collect_charuco_paired(
            left_paths, right_paths, board, dictionary,
            min_markers=min_markers, min_charuco=min_charuco,
        )
    )

    if report_json is None:
        report_json = out_yaml.with_suffix(".report.json")

    # ✅ 先寫 report（就算後面 fail 也會留下）
    report = {
        "status": "collected",
        "left": l_report.__dict__,
        "right": r_report.__dict__,
        "inputs": {
            "data_dir": str(data_dir),
            "num_left_images": len(left_paths),
            "num_right_images": len(right_paths),
        },
        "params": {
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length": square_length,
            "marker_length": marker_length,
            "dict_name": dict_name,
            "min_views": min_views,
            "min_markers": min_markers,
            "min_charuco": min_charuco,
            "min_common_ids": min_common_ids,
        },
        "precheck": {
            "paired_valid_views_est": min(l_report.ok, r_report.ok),
            "image_size": {"width": img_size[0], "height": img_size[1]},
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ✅ 在進 stereo calibration 前就先 fail-fast（否則你會永遠看不到 report）
    n = min(len(l_corners), len(r_corners))
    if n < min_views:
        report["status"] = "failed"
        report["reason"] = "not_enough_valid_views"
        report["precheck"]["paired_valid_views_est"] = n
        report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Not enough valid paired views. got={n}, need>={min_views}. "
            f"See report: {report_json}"
        )

    # ✅ 真的開始 stereo calibration
    result = run_stereo_calibration(
        l_corners, l_ids, r_corners, r_ids, img_size, board,
        min_views=min_views, min_common_ids=min_common_ids
    )

    calib_dict = asdict(result)
    calib_out = {
        "image_size": {"width": calib_dict["image_size"][0], "height": calib_dict["image_size"][1]},
        "K1": calib_dict["K1"], "D1": calib_dict["D1"],
        "K2": calib_dict["K2"], "D2": calib_dict["D2"],
        "R": calib_dict["R"], "T": calib_dict["T"],
        "baseline_m": calib_dict["baseline_m"],
        "R1": calib_dict["R1"], "R2": calib_dict["R2"],
        "P1": calib_dict["P1"], "P2": calib_dict["P2"],
        "Q": calib_dict["Q"],
        "metrics": {
            "mono_reproj_L": calib_dict["mono_reproj_L"],
            "mono_reproj_R": calib_dict["mono_reproj_R"],
            "stereo_rms": calib_dict["stereo_rms"],
            "used_views": calib_dict["used_views"],
            "matched_views": calib_dict["matched_views"],
        },
    }

    save_yaml(out_yaml, calib_out)

    # ✅ 更新 report 為成功
    report["status"] = "success"
    report["metrics"] = calib_out["metrics"]
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if per_image_rpe:
        _print_per_image_rpe(
            result, l_corners, l_ids, r_corners, r_ids,
            matched_left_names, board, min_common_ids,
        )

    return out_yaml, report_json


def _print_per_image_rpe(
    result,
    l_corners, l_ids,
    r_corners, r_ids,
    matched_left_names: list[str],
    board,
    min_common_ids: int,
) -> None:
    """Compute and print per-image stereo RPE sorted worst to best."""
    from stereo_depth.adapters.calibration.charuco_calibrator import _match_ids_one_view

    K1 = np.array(result.K1, dtype=np.float64)
    D1 = np.array(result.D1, dtype=np.float64)
    K2 = np.array(result.K2, dtype=np.float64)
    D2 = np.array(result.D2, dtype=np.float64)
    R  = np.array(result.R,  dtype=np.float64)
    T  = np.array(result.T,  dtype=np.float64)

    chess_corners_3d = board.getChessboardCorners()
    n = result.used_views
    lc_used = l_corners[:n]
    li_used = l_ids[:n]
    rc_used = r_corners[:n]
    ri_used = r_ids[:n]

    # Re-run the ID-matching to recover objpoints/imgpoints (same logic as
    # run_stereo_calibration) while keeping track of which name each view has.
    objpoints:  list[np.ndarray] = []
    imgpointsL: list[np.ndarray] = []
    imgpointsR: list[np.ndarray] = []
    names:      list[str]        = []

    for lc, li, rc, ri, name in zip(lc_used, li_used, rc_used, ri_used, matched_left_names):
        matched = _match_ids_one_view(lc, li, rc, ri, chess_corners_3d,
                                      min_common=min_common_ids)
        if matched is None:
            continue
        obj, ptsL, ptsR = matched
        objpoints.append(obj)
        imgpointsL.append(ptsL)
        imgpointsR.append(ptsR)
        names.append(name)

    rpes = compute_per_image_rpe(objpoints, imgpointsL, imgpointsR, K1, D1, K2, D2, R, T)

    pairs = sorted(zip(rpes, names), reverse=True)

    print("\nPer-image RPE (sorted worst → best):")
    for rpe, name in pairs:
        if np.isfinite(rpe):
            print(f"  {name:<30s}  {rpe:.3f} px")
        else:
            print(f"  {name:<30s}  N/A")
