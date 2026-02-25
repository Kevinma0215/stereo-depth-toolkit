"""verify_depth_accuracy.py — manual tool for measuring stereo depth pipeline error.

Usage
-----
python tools/verify_depth_accuracy.py \
    --calib  outputs/calib/calib_strict.yaml \
    --left   data/calib/charuco_2026-02-14_run1/left/left_00007.png \
    --right  data/calib/charuco_2026-02-14_run1/right/right_00007.png \
    --distance 0.5

Optionally restrict the measurement to a specific image region:
    --roi 600,200,320,240    # x,y,w,h  (top-left corner, width, height)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Library imports (direct, not via CLI)
# ---------------------------------------------------------------------------
from stereo_depth.adapters.calibration.yaml_repo import YamlCalibrationRepo
from stereo_depth.adapters.camera.file_source import FileSource
from stereo_depth.adapters.depth.opencv_depth_estimator import OpencvDepthEstimator
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.use_cases.pipeline import StereoPipeline

OUTPUT_DIR = Path("outputs/depth_verify")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be four comma-separated integers: x,y,w,h")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("--roi values must be integers")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("--roi width and height must be > 0")
    return x, y, w, h


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify stereo depth pipeline accuracy against a known ground-truth distance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--calib",    required=True,  metavar="PATH",  help="calibration YAML file")
    p.add_argument("--left",     required=True,  metavar="PATH",  help="left image file")
    p.add_argument("--right",    required=True,  metavar="PATH",  help="right image file")
    p.add_argument("--distance", required=True,  type=float,      metavar="M",
                   help="ground-truth distance to the target object, in metres")
    p.add_argument("--roi",      default=None,   type=_parse_roi, metavar="x,y,w,h",
                   help="measurement region (pixels). Defaults to central 10%% of the image.")
    p.add_argument("--out",      default=str(OUTPUT_DIR), metavar="DIR",
                   help=f"output directory for the debug image (default: {OUTPUT_DIR})")
    p.add_argument("--preset",   default="indoor",
                   choices=("indoor", "outdoor", "high_quality"),
                   help="SGBM preset name (default: indoor)")
    return p


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _central_roi(h: int, w: int, fraction: float = 0.10) -> tuple[int, int, int, int]:
    """Return (x, y, roi_w, roi_h) covering the central `fraction` of the image."""
    roi_w = max(1, int(w * fraction))
    roi_h = max(1, int(h * fraction))
    x = (w - roi_w) // 2
    y = (h - roi_h) // 2
    return x, y, roi_w, roi_h


def _crop_roi(arr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return arr[y : y + h, x : x + w]


def _disparity_colormap(disp: np.ndarray) -> np.ndarray:
    """Convert float32 disparity to an 8-bit BGR colourmap image."""
    valid = disp[disp > 0]
    if valid.size == 0:
        return np.zeros((*disp.shape, 3), dtype=np.uint8)
    lo, hi = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    span = hi - lo if hi > lo else 1.0
    norm = np.clip((disp - lo) / span, 0.0, 1.0)
    grey = (norm * 255).astype(np.uint8)
    grey[disp <= 0] = 0
    return cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)


def _save_debug_image(
    left_rect: np.ndarray,
    disparity: np.ndarray,
    roi: tuple[int, int, int, int],
    median_depth: float,
    out_path: Path,
) -> None:
    x, y, w, h = roi

    # Left rectified with green ROI box
    vis_left = left_rect.copy() if left_rect.ndim == 3 else cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_left, (x, y), (x + w, y + h), (0, 220, 0), 2)
    label = f"median {median_depth:.3f} m"
    cv2.putText(vis_left, label, (x, max(y - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)

    # Disparity colourmap (same ROI box)
    disp_color = _disparity_colormap(disparity)
    cv2.rectangle(disp_color, (x, y), (x + w, y + h), (0, 220, 0), 2)

    # Side-by-side
    H = max(vis_left.shape[0], disp_color.shape[0])

    def _pad(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return np.vstack([img, pad])

    combined = np.hstack([_pad(vis_left, H), _pad(disp_color, H)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


def run(args: argparse.Namespace) -> None:
    # ---------- build pipeline ----------
    calib = YamlCalibrationRepo().load(args.calib)
    source = FileSource(args.left, args.right)

    pipeline = StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=SgbmMatcher(preset=args.preset),
        depth_estimator=OpencvDepthEstimator(),
        calib=calib,
    )

    depth_map = pipeline.process(source.grab())
    depth = depth_map.data          # float32 (H, W), metres, NaN = invalid
    disp  = depth_map.disparity
    left_rect = depth_map.left_rect  # may be None if not attached

    H, W = depth.shape

    # ---------- determine ROI ----------
    if args.roi is not None:
        roi = args.roi
        x, y, w, h = roi
        # clamp to image bounds
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        roi = (x, y, w, h)
    else:
        roi = _central_roi(H, W)

    roi_depth = _crop_roi(depth, roi).ravel()

    # ---------- statistics (exclude NaN and inf) ----------
    finite = roi_depth[np.isfinite(roi_depth)]
    total_px = roi_depth.size

    if finite.size == 0:
        print("ERROR: no valid (finite) depth pixels in the specified ROI.", file=sys.stderr)
        print("  Try a different --roi, or check the calibration and images.")
        sys.exit(1)

    median_depth = float(np.median(finite))
    mean_depth   = float(np.mean(finite))
    std_depth    = float(np.std(finite))
    valid_ratio  = finite.size / total_px
    abs_error    = abs(median_depth - args.distance)
    rel_error    = abs_error / args.distance * 100.0

    # ---------- terminal report ----------
    SEP = "-" * 36
    print(SEP)
    print(f"{'實際距離':12s}  {args.distance:.3f} m")
    print(f"{'中位數深度':12s}  {median_depth:.3f} m")
    print(f"{'平均深度':12s}  {mean_depth:.3f} m")
    print(f"{'標準差':12s}  {std_depth:.3f} m")
    print(f"{'有效像素':12s}  {valid_ratio * 100:.1f}%")
    print(f"{'絕對誤差':12s}  {abs_error:.3f} m")
    print(f"{'相對誤差':12s}  {rel_error:.1f}%")
    print(SEP)

    # ---------- debug image ----------
    ref_img = (
        left_rect
        if left_rect is not None
        else cv2.imread(str(args.left))
    )
    if ref_img is None:
        ref_img = np.zeros((H, W, 3), dtype=np.uint8)

    out_dir  = Path(args.out)
    stem     = Path(args.left).stem
    out_path = out_dir / f"verify_{stem}.png"
    _save_debug_image(ref_img, disp, roi, median_depth, out_path)
    print(f"debug 圖已儲存：{out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_parser()
    run(parser.parse_args())
