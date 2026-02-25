"""robot_arm_stream.py — example: feed a live stereo stream into a robot-arm controller.

Real camera usage (requires /dev/video0 + calibration YAML):

    from stereo_depth.adapters.camera.uvc_source import UvcSource
    from stereo_depth.adapters.calibration.yaml_repo import YamlCalibrationRepo

    calib = YamlCalibrationRepo().load("outputs/calib/calib.yaml")
    pipeline = StereoPipeline(
        camera_source=UvcSource(device_index=0, width=2560, height=720),
        rectifier=OpenCVRectifier(),
        matcher=SgbmMatcher(preset="indoor"),
        depth_estimator=OpencvDepthEstimator(),
        calib=calib,
    )
    for depth_map in pipeline.stream():
        # plug your robot-arm logic here
        ...

This script runs in mock mode by default (no camera needed) so it can be
verified in CI or on a development machine without attached hardware.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from stereo_depth.adapters.camera.file_source import FileSource
from stereo_depth.adapters.depth.opencv_depth_estimator import OpencvDepthEstimator
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.entities import CalibrationResult
from stereo_depth.use_cases.pipeline import StereoPipeline

# ---------------------------------------------------------------------------
# Mock calibration — identity-like matrices valid for a (320 × 180) image
# ---------------------------------------------------------------------------

def _mock_calib(w: int = 320, h: int = 180, baseline: float = 0.06) -> CalibrationResult:
    f = float(w)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    T = np.array([-baseline, 0.0, 0.0], dtype=np.float64)
    # Rectification maps: identity (images are already "rectified" in mock)
    R1 = R2 = np.eye(3, dtype=np.float64)
    P1 = np.hstack([K, np.zeros((3, 1), dtype=np.float64)])
    P2 = P1.copy()
    P2[0, 3] = -f * baseline
    Q = np.array([
        [1, 0,  0,  -cx],
        [0, 1,  0,  -cy],
        [0, 0,  0,    f],
        [0, 0, -1/baseline, 0],
    ], dtype=np.float64)
    return CalibrationResult(
        image_size=(w, h),
        K1=K, D1=D, K2=K, D2=D,
        R=R, T=T, baseline_m=baseline,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        rpe_px=0.0,
    )


# ---------------------------------------------------------------------------
# Mock image pair generator — synthetic texture with a known disparity shift
# ---------------------------------------------------------------------------

def _write_mock_frames(
    directory: Path, n: int = 5, w: int = 320, h: int = 180, shift: int = 20
) -> None:
    left_dir = directory / "left"
    right_dir = directory / "right"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(n):
        left = rng.integers(30, 220, (h, w, 3), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, : w - shift] = left[:, shift:]
        cv2.imwrite(str(left_dir / f"{i:04d}.png"), left)
        cv2.imwrite(str(right_dir / f"{i:04d}.png"), right)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    N_FRAMES = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_mock_frames(tmp, n=N_FRAMES)

        calib = _mock_calib()
        pipeline = StereoPipeline(
            camera_source=FileSource(tmp / "left", tmp / "right"),
            rectifier=OpenCVRectifier(),
            matcher=SgbmMatcher(preset="indoor"),
            depth_estimator=OpencvDepthEstimator(),
            calib=calib,
        )

        for i, depth_map in enumerate(pipeline.stream()):
            valid = depth_map.data[~np.isnan(depth_map.data)]
            d_min = float(np.min(valid)) if valid.size else float("nan")
            d_max = float(np.max(valid)) if valid.size else float("nan")
            print(
                f"[frame {i:02d}] depth shape: {depth_map.data.shape}  "
                f"valid px: {valid.size}  "
                f"min: {d_min:.2f} m  max: {d_max:.2f} m"
            )


if __name__ == "__main__":
    main()
