"""把「收集→校正→輸出→report」串起來（report 永遠產生）"""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json

from stereo_depth.calib.boards import make_charuco_board
from stereo_depth.calib.collect import collect_charuco_from_paths
from stereo_depth.calib.stereo_calib import run_stereo_calibration
from stereo_depth.config.io import save_yaml


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

    # 這裡不強制 left/right 數量相等：先各自收集，再用 paired ok 數量決定
    l_corners, l_ids, img_size, l_report = collect_charuco_from_paths(
        left_paths, board, dictionary, min_markers=min_markers, min_charuco=min_charuco
    )
    r_corners, r_ids, _, r_report = collect_charuco_from_paths(
        right_paths, board, dictionary, min_markers=min_markers, min_charuco=min_charuco
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

    return out_yaml, report_json
