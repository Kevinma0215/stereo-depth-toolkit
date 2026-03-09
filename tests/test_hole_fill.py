"""Tests for OpenCVDepthEstimator optional hole-filling (cv2.inpaint INPAINT_NS)."""
from __future__ import annotations
import numpy as np
import pytest

from stereo_depth.adapters.depth.opencv_depth_estimator import OpenCVDepthEstimator
from stereo_depth.entities import CalibrationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calib() -> CalibrationResult:
    """Minimal CalibrationResult with a valid Q matrix (fx=480, B=0.06 m)."""
    fx = 480.0
    cx, cy = 320.0, 240.0
    Tx = -fx * 0.06  # -28.8

    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0,  fx],
        [0, 0, 1 / 0.06, 0],
    ], dtype=np.float64)

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = fx
    K[0, 2] = cx
    K[1, 2] = cy

    return CalibrationResult(
        image_size=(640, 480),
        K1=K, D1=np.zeros(5),
        K2=K, D2=np.zeros(5),
        R=np.eye(3), T=np.array([-0.06, 0.0, 0.0]),
        baseline_m=0.06,
        R1=np.eye(3), R2=np.eye(3),
        P1=np.hstack([K, np.zeros((3, 1))]),
        P2=np.hstack([K, np.array([[Tx], [0], [0]])]),
        Q=Q,
        rpe_px=0.3,
    )


def _flat_disparity(h: int = 60, w: int = 80, value: float = 24.0) -> np.ndarray:
    """All-valid flat disparity map."""
    return np.full((h, w), value, dtype=np.float32)


def _disparity_with_holes(h: int = 60, w: int = 80, hole_rows=slice(20, 40), hole_cols=slice(30, 50)) -> np.ndarray:
    """Disparity map with a rectangular hole (0 = invalid)."""
    d = _flat_disparity(h, w)
    d[hole_rows, hole_cols] = 0.0
    return d


# ---------------------------------------------------------------------------
# Tests: fill_holes=False (default)
# ---------------------------------------------------------------------------

class TestNoFill:
    def test_default_constructor_no_fill(self):
        est = OpenCVDepthEstimator()
        assert est._fill_holes is False

    def test_invalid_pixels_are_nan_without_fill(self):
        calib = _make_calib()
        est = OpenCVDepthEstimator(fill_holes=False)
        disp = _disparity_with_holes()
        dm = est.to_depth(disp, calib)
        assert np.any(~np.isfinite(dm.data)), "holes should remain NaN when fill_holes=False"

    def test_valid_pixels_unchanged_without_fill(self):
        calib = _make_calib()
        est = OpenCVDepthEstimator(fill_holes=False)
        disp = _flat_disparity()
        dm = est.to_depth(disp, calib)
        assert np.all(np.isfinite(dm.data)), "all pixels should be finite for hole-free input"

    def test_disparity_preserved_in_output(self):
        calib = _make_calib()
        est = OpenCVDepthEstimator(fill_holes=False)
        disp = _flat_disparity()
        dm = est.to_depth(disp, calib)
        np.testing.assert_array_equal(dm.disparity, disp)


# ---------------------------------------------------------------------------
# Tests: fill_holes=True
# ---------------------------------------------------------------------------

class TestWithFill:
    def test_fill_reduces_invalid_count(self):
        calib = _make_calib()
        disp = _disparity_with_holes()

        est_no_fill = OpenCVDepthEstimator(fill_holes=False)
        est_fill = OpenCVDepthEstimator(fill_holes=True)

        dm_no = est_no_fill.to_depth(disp, calib)
        dm_yes = est_fill.to_depth(disp, calib)

        invalid_before = int((~np.isfinite(dm_no.data)).sum())
        invalid_after = int((~np.isfinite(dm_yes.data)).sum())
        assert invalid_after < invalid_before, (
            f"fill_holes=True should reduce invalid count: before={invalid_before}, after={invalid_after}"
        )

    def test_filled_depth_plausible(self):
        """Filled values should be in (0, 20] m — not wildly wrong."""
        calib = _make_calib()
        disp = _disparity_with_holes()
        est = OpenCVDepthEstimator(fill_holes=True)
        dm = est.to_depth(disp, calib)
        finite = dm.data[np.isfinite(dm.data)]
        assert np.all(finite > 0), "filled depth must be positive"
        assert np.all(finite <= 20.0), "filled depth must be <= 20 m (scale limit)"

    def test_valid_pixels_unmodified_by_fill(self):
        """Pixels that were valid before filling must keep the same depth value."""
        calib = _make_calib()
        disp = _disparity_with_holes()

        est_no_fill = OpenCVDepthEstimator(fill_holes=False)
        est_fill = OpenCVDepthEstimator(fill_holes=True)

        dm_ref = est_no_fill.to_depth(disp, calib)
        dm_filled = est_fill.to_depth(disp, calib)

        valid_mask = np.isfinite(dm_ref.data)
        np.testing.assert_allclose(
            dm_filled.data[valid_mask],
            dm_ref.data[valid_mask],
            rtol=0,
            atol=0,
            err_msg="fill_holes=True must not alter originally-valid pixels",
        )

    def test_fill_radius_larger_fills_more(self):
        """A larger fill_radius should fill at least as many pixels as a smaller one."""
        calib = _make_calib()
        disp = _disparity_with_holes(h=80, w=80, hole_rows=slice(30, 50), hole_cols=slice(30, 50))

        def count_invalid(radius):
            dm = OpenCVDepthEstimator(fill_holes=True, fill_radius=radius).to_depth(disp, calib)
            return int((~np.isfinite(dm.data)).sum())

        assert count_invalid(10) <= count_invalid(1)

    def test_all_holes_disparity_returns_all_nan_without_fill(self):
        calib = _make_calib()
        disp = np.zeros((30, 30), dtype=np.float32)  # all invalid
        est = OpenCVDepthEstimator(fill_holes=False)
        dm = est.to_depth(disp, calib)
        assert np.all(~np.isfinite(dm.data))

    def test_all_holes_fill_produces_output(self):
        """Even an all-invalid disparity map should return *some* finite depth after fill."""
        calib = _make_calib()
        disp = np.zeros((30, 30), dtype=np.float32)
        est = OpenCVDepthEstimator(fill_holes=True)
        dm = est.to_depth(disp, calib)
        # inpaint with no valid source pixels produces zeros → 0/scale = 0.0 (still finite)
        assert np.all(np.isfinite(dm.data)), "all-hole input should produce finite (possibly 0) depth after fill"


# ---------------------------------------------------------------------------
# Tests: constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_fill_radius_zero_raises(self):
        with pytest.raises(ValueError, match="fill_radius"):
            OpenCVDepthEstimator(fill_radius=0)

    def test_fill_radius_negative_raises(self):
        with pytest.raises(ValueError, match="fill_radius"):
            OpenCVDepthEstimator(fill_radius=-5)

    def test_fill_radius_one_valid(self):
        est = OpenCVDepthEstimator(fill_holes=True, fill_radius=1)
        assert est._fill_radius == 1
