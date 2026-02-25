"""End-to-end integration test for StereoPipeline.

Uses real calibration data (outputs/calib/calib.yaml) and a real image pair
from the ChArUco calibration session.  No camera required.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from stereo_depth.infrastructure.config.io import load_yaml
from stereo_depth.entities import CalibrationResult, DepthMap
from stereo_depth.adapters.camera.file_source import FileSource
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.adapters.depth.opencv_depth_estimator import OpenCVDepthEstimator
from stereo_depth.use_cases.pipeline import StereoPipeline

# ---------------------------------------------------------------------------
# Paths (relative to repo root; pytest is run from there)
# ---------------------------------------------------------------------------
CALIB_YAML = Path("outputs/calib/calib.yaml")
DATA_DIR = Path("data/calib/charuco_2026-02-14_run1")
LEFT_IMG = DATA_DIR / "left" / "left_00007.png"
RIGHT_IMG = DATA_DIR / "right" / "right_00007.png"


# ---------------------------------------------------------------------------
# Helper: YAML dict → CalibrationResult
# ---------------------------------------------------------------------------

def _load_calib(path: Path) -> CalibrationResult:
    data = load_yaml(path)
    w = int(data["image_size"]["width"])
    h = int(data["image_size"]["height"])
    # rpe_px not yet in the existing YAML; fall back to stereo_rms from metrics
    rpe = float(
        data.get("rpe_px", data.get("metrics", {}).get("stereo_rms", 0.0))
    )
    return CalibrationResult(
        image_size=(w, h),
        K1=np.array(data["K1"], dtype=np.float64),
        D1=np.array(data["D1"], dtype=np.float64),
        K2=np.array(data["K2"], dtype=np.float64),
        D2=np.array(data["D2"], dtype=np.float64),
        R=np.array(data["R"], dtype=np.float64),
        T=np.array(data["T"], dtype=np.float64),
        baseline_m=float(data["baseline_m"]),
        R1=np.array(data["R1"], dtype=np.float64),
        R2=np.array(data["R2"], dtype=np.float64),
        P1=np.array(data["P1"], dtype=np.float64),
        P2=np.array(data["P2"], dtype=np.float64),
        Q=np.array(data["Q"], dtype=np.float64),
        rpe_px=rpe,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calib() -> CalibrationResult:
    if not CALIB_YAML.exists():
        pytest.skip(f"calibration file not found: {CALIB_YAML}")
    return _load_calib(CALIB_YAML)


@pytest.fixture(scope="module")
def image_pair_exists() -> bool:
    return LEFT_IMG.exists() and RIGHT_IMG.exists()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_output_shape(calib, image_pair_exists):
    """Full pipeline must return a DepthMap with the correct spatial shape."""
    if not image_pair_exists:
        pytest.skip(f"image pair not found: {LEFT_IMG}, {RIGHT_IMG}")

    source = FileSource(LEFT_IMG, RIGHT_IMG)
    pipeline = StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=SgbmMatcher(preset_name="indoor"),
        depth_estimator=OpenCVDepthEstimator(),
        calib=calib,
    )

    depth_map = pipeline.process(source.grab())

    w, h = calib.image_size
    assert isinstance(depth_map, DepthMap)
    assert depth_map.data.shape == (h, w), (
        f"expected depth shape ({h}, {w}), got {depth_map.data.shape}"
    )
    assert depth_map.disparity.shape == (h, w)
    assert depth_map.data.dtype == np.float32
    assert depth_map.disparity.dtype == np.float32


def test_pipeline_depth_not_all_nan(calib, image_pair_exists):
    """At least some pixels must have a valid (non-NaN) depth value."""
    if not image_pair_exists:
        pytest.skip(f"image pair not found: {LEFT_IMG}, {RIGHT_IMG}")

    source = FileSource(LEFT_IMG, RIGHT_IMG)
    pipeline = StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=SgbmMatcher(preset_name="indoor"),
        depth_estimator=OpenCVDepthEstimator(),
        calib=calib,
    )

    depth_map = pipeline.process(source.grab())

    valid = depth_map.data[~np.isnan(depth_map.data)]
    assert valid.size > 0, "all depth pixels are NaN — pipeline produced no valid depth"


def test_pipeline_left_rect_attached(calib, image_pair_exists):
    """StereoPipeline must attach the rectified left image to the DepthMap."""
    if not image_pair_exists:
        pytest.skip(f"image pair not found: {LEFT_IMG}, {RIGHT_IMG}")

    source = FileSource(LEFT_IMG, RIGHT_IMG)
    pipeline = StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=SgbmMatcher(preset_name="indoor"),
        depth_estimator=OpenCVDepthEstimator(),
        calib=calib,
    )

    depth_map = pipeline.process(source.grab())

    w, h = calib.image_size
    assert depth_map.left_rect is not None
    assert depth_map.left_rect.shape == (h, w, 3)
