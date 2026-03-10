"""tools/evaluate_depth.py — evaluate depth accuracy against recorded sequences.

Loads a sequence captured by capture_accuracy_seq.py, runs process_stack(),
samples a 20×20 patch at the frame centre, and reports accuracy vs. the
ground-truth distance stored in metadata.yaml.

Usage:
    # Single sequence
    python tools/evaluate_depth.py --seq-dir data/accuracy/dist_030cm/

    # All sequences under data/accuracy/
    python tools/evaluate_depth.py --all

Exit code:
    0 — all evaluated sequences PASS
    1 — one or more FAIL (usable as a CI gate)

PASS criteria:
    error_pct  < 5 %   (< 10 % for distances <= 0.25 m)
    AND coverage_pct > 80 %
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from stereo_depth.adapters.calibration.yaml_repo import YamlCalibrationRepo
from stereo_depth.adapters.camera.file_source import FileSource
from stereo_depth.adapters.depth.opencv_depth_estimator import OpenCVDepthEstimator
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.infrastructure.config.io import load_yaml, save_yaml
from stereo_depth.use_cases.pipeline import StereoPipeline


_DEFAULT_DATA_DIR   = Path("data/accuracy")
_PATCH_HALF         = 10    # 20×20 patch = centre ± 10 px
_COVERAGE_THRESHOLD = 80.0  # %


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def _error_threshold(distance_m: float) -> float:
    """Return the PASS error threshold (%) for the given distance."""
    return 10.0 if distance_m <= 0.25 else 5.0


def _is_pass(error_pct: float, coverage_pct: float, distance_m: float) -> bool:
    return (
        np.isfinite(error_pct)
        and error_pct    < _error_threshold(distance_m)
        and coverage_pct > _COVERAGE_THRESHOLD
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate stereo depth accuracy against sequences recorded "
                    "by capture_accuracy_seq.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--seq-dir", type=Path, metavar="DIR",
        help="Path to a single sequence directory (must contain metadata.yaml).",
    )
    group.add_argument(
        "--all", action="store_true",
        help=f"Evaluate every sequence found under --data-dir "
             f"(default: {_DEFAULT_DATA_DIR}/).",
    )
    p.add_argument(
        "--data-dir", type=Path, default=_DEFAULT_DATA_DIR, metavar="DIR",
        help="Root data directory searched when --all is used; "
             "also controls where results_*.yaml is written.",
    )
    p.add_argument(
        "--calib", default=None, metavar="PATH",
        help="Override the calibration YAML for all sequences "
             "(default: read calib_path from each metadata.yaml).",
    )
    p.add_argument(
        "--preset", default="indoor",
        choices=("indoor", "outdoor", "high_quality"),
        help="SGBM preset name.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sequence loading
# ---------------------------------------------------------------------------

def _find_sequences(data_dir: Path) -> list[Path]:
    """Return subdirectories of data_dir that contain a metadata.yaml."""
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and (p / "metadata.yaml").exists()
    )


def _load_sbs_frames(seq_dir: Path) -> list[np.ndarray]:
    """Load every stereo pair from seq_dir/left + seq_dir/right.

    FileSource (directory mode) yields FramePairs matched by sorted filename.
    Each SBS frame is reassembled as np.hstack([pair.left, pair.right]),
    producing the (H, W*2, 3) uint8 shape that process_stack() expects.
    """
    source = FileSource(seq_dir / "left", seq_dir / "right")
    frames: list[np.ndarray] = []
    for pair in source.stream():
        frames.append(np.hstack([pair.left, pair.right]))
    return frames


def _center_patch(depth: np.ndarray, half: int = _PATCH_HALF) -> np.ndarray:
    """Return the (2*half) × (2*half) patch centred in depth."""
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    return depth[cy - half : cy + half, cx - half : cx + half]


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def _evaluate_sequence(
    seq_dir: Path,
    calib_override: str | None,
    preset: str,
) -> dict:
    """Run process_stack on one sequence and return metrics."""
    meta       = load_yaml(seq_dir / "metadata.yaml")
    distance_m = float(meta["distance_m"])
    calib_path = calib_override or str(meta["calib_path"])

    frames = _load_sbs_frames(seq_dir)
    if not frames:
        raise RuntimeError(f"No images found in {seq_dir}/left or {seq_dir}/right")

    calib = YamlCalibrationRepo().load(calib_path)
    pipeline = StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=SgbmMatcher(preset_name=preset),
        depth_estimator=OpenCVDepthEstimator(),
        calib=calib,
    )

    # process_stack expects temporal_frames == len(frames)
    depth = pipeline.process_stack(frames, temporal_frames=len(frames))

    patch       = _center_patch(depth)
    total_px    = patch.size
    valid_px    = int(np.sum(~np.isnan(patch)))
    coverage_pct = valid_px / total_px * 100.0

    measured_m = float(np.nanmedian(patch)) if valid_px > 0 else float("nan")
    error_pct  = (
        abs(measured_m - distance_m) / distance_m * 100.0
        if np.isfinite(measured_m) else float("nan")
    )
    passed = _is_pass(error_pct, coverage_pct, distance_m)

    return {
        "distance_m":   distance_m,
        "measured_m":   measured_m,
        "error_pct":    error_pct,
        "coverage_pct": coverage_pct,
        "pass":         passed,
        "calib_path":   calib_path,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt_f(v: float, fmt: str) -> str:
    return format(v, fmt) if np.isfinite(v) else "N/A"


def _print_table(results: list[dict]) -> None:
    col_w = [8, 12, 9, 12, 9]
    header = (
        f"{'dist (m)':>{col_w[0]}} | "
        f"{'measured (m)':>{col_w[1]}} | "
        f"{'error (%)':>{col_w[2]}} | "
        f"{'coverage (%)':>{col_w[3]}} | "
        f"PASS/FAIL"
    )
    sep = (
        "-" * (col_w[0] + 1) + "+"
        + "-" * (col_w[1] + 2) + "+"
        + "-" * (col_w[2] + 2) + "+"
        + "-" * (col_w[3] + 2) + "+"
        + "-" * 9
    )
    print()
    print(header)
    print(sep)
    for r in results:
        verdict  = "PASS" if r["pass"] else "FAIL"
        measured = _fmt_f(r["measured_m"],   ".4f")
        error    = _fmt_f(r["error_pct"],    ".2f")
        coverage = _fmt_f(r["coverage_pct"], ".1f")
        print(
            f"{r['distance_m']:>{col_w[0]}.2f} | "
            f"{measured:>{col_w[1]}} | "
            f"{error:>{col_w[2]}} | "
            f"{coverage:>{col_w[3]}} | "
            f"{verdict}"
        )
    print()


def _results_for_yaml(results: list[dict]) -> list[dict]:
    """Convert float NaN to None so yaml.safe_dump doesn't choke."""
    out = []
    for r in results:
        out.append({
            "distance_m":   r["distance_m"],
            "measured_m":   None if not np.isfinite(r["measured_m"]) else r["measured_m"],
            "error_pct":    None if not np.isfinite(r["error_pct"])  else r["error_pct"],
            "coverage_pct": r["coverage_pct"],
            "pass":         r["pass"],
        })
    return out


def to_python(obj):
    """Recursively convert numpy scalar types to native Python types."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Collect sequence directories
    if args.all:
        seq_dirs = _find_sequences(args.data_dir)
        if not seq_dirs:
            print(f"No sequences found under {args.data_dir}/", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.seq_dir.exists():
            print(f"ERROR: {args.seq_dir} does not exist.", file=sys.stderr)
            sys.exit(1)
        seq_dirs = [args.seq_dir]

    # Evaluate
    results: list[dict] = []
    skipped: list[str]  = []

    for seq_dir in seq_dirs:
        try:
            print(f"  Evaluating {seq_dir.name} ...", end=" ", flush=True)
            result = _evaluate_sequence(seq_dir, args.calib, args.preset)
            results.append(result)
            print("done")
        except Exception as exc:
            skipped.append(f"{seq_dir.name}: {exc}")
            print(f"SKIP ({exc})")

    if not results:
        print("No sequences could be evaluated.", file=sys.stderr)
        sys.exit(1)

    # Print table
    _print_table(results)

    # Summary line
    all_pass = all(r["pass"] for r in results)
    n_pass   = sum(1 for r in results if r["pass"])
    print(f"  {n_pass}/{len(results)} sequences PASS")
    if skipped:
        print(f"  {len(skipped)} sequence(s) skipped due to errors:")
        for msg in skipped:
            print(f"    {msg}")
    print()

    # Save results YAML
    calib_used = args.calib or results[0]["calib_path"]
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_yaml   = args.data_dir / f"results_{ts}.yaml"
    save_yaml(out_yaml, to_python({
        "results":    _results_for_yaml(results),
        "all_pass":   all_pass,
        "timestamp":  datetime.now().isoformat(),
        "calib_path": calib_used,
    }))
    print(f"Results saved to {out_yaml}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
