"""Tests for evaluate_calibration().

All tests are hardware-independent: calibrations are synthesised with
cv2.stereoRectify so no real camera or saved data is needed.
"""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from stereo_depth.entities import CalibrationResult
from stereo_depth.adapters.calibration.eval_calibration import (
    CalibrationEvaluation,
    evaluate_calibration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_good_calib(
    w: int = 640,
    h: int = 480,
    f: float = 500.0,
    baseline: float = 0.06,
    rot_deg: float = 2.0,
    rpe_px: float = 0.3,
) -> CalibrationResult:
    """Synthetically correct stereo calibration built via cv2.stereoRectify.

    The R1/R2/P1/P2 matrices are computed by OpenCV's own rectification
    algorithm, so the resulting epipolar error is at floating-point precision
    (< 0.01 px).
    """
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)

    angle = np.deg2rad(rot_deg)
    R = np.array(
        [
            [ np.cos(angle), 0, np.sin(angle)],
            [0,              1, 0             ],
            [-np.sin(angle), 0, np.cos(angle) ],
        ],
        dtype=np.float64,
    )
    T = np.array([-baseline, 0.0, 0.0], dtype=np.float64)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, D, K, D, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    return CalibrationResult(
        image_size=(w, h),
        K1=K.copy(), D1=D.copy(),
        K2=K.copy(), D2=D.copy(),
        R=R, T=T,
        baseline_m=float(np.linalg.norm(T)),
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        rpe_px=rpe_px,
    )


def _make_bad_rectification_calib(rot_deg: float = 5.0) -> CalibrationResult:
    """CalibrationResult with a real inter-camera rotation but wrong (identity)
    R1/R2.  The epipolar lines are therefore not aligned, producing a large
    epipolar error.
    """
    w, h = 640, 480
    f = 500.0
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)

    angle = np.deg2rad(rot_deg)
    R = np.array(
        [
            [ np.cos(angle), 0, np.sin(angle)],
            [0,              1, 0             ],
            [-np.sin(angle), 0, np.cos(angle) ],
        ],
        dtype=np.float64,
    )
    T = np.array([-0.06, 0.0, 0.0], dtype=np.float64)

    # Intentionally wrong: identity R1/R2 instead of the stereoRectify output
    R1 = np.eye(3, dtype=np.float64)
    R2 = np.eye(3, dtype=np.float64)
    P1 = np.hstack([K, np.zeros((3, 1), dtype=np.float64)])
    P2 = np.hstack([K, np.zeros((3, 1), dtype=np.float64)])
    Q  = np.eye(4, dtype=np.float64)  # dummy

    return CalibrationResult(
        image_size=(w, h),
        K1=K, D1=D, K2=K, D2=D,
        R=R, T=T,
        baseline_m=float(np.linalg.norm(T)),
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        rpe_px=0.0,
    )


# ---------------------------------------------------------------------------
# Tests — return type and field completeness
# ---------------------------------------------------------------------------

def test_returns_calibration_evaluation():
    """evaluate_calibration() must return a CalibrationEvaluation instance."""
    calib = _make_good_calib()
    result = evaluate_calibration(calib)
    assert isinstance(result, CalibrationEvaluation)


def test_all_fields_present_and_finite():
    """Every numeric field must be a finite float; warnings must be a list."""
    calib = _make_good_calib()
    ev = evaluate_calibration(calib)

    for field_name in (
        "rpe_px",
        "epipolar_error_mean_px",
        "epipolar_error_max_px",
        "focal_length_asymmetry",
        "principal_point_offset_px",
        "baseline_m",
    ):
        value = getattr(ev, field_name)
        assert isinstance(value, float), f"{field_name} is not float"
        assert np.isfinite(value),       f"{field_name} is not finite"

    assert isinstance(ev.passed,   bool)
    assert isinstance(ev.warnings, list)


# ---------------------------------------------------------------------------
# Tests — correct calibration passes all checks
# ---------------------------------------------------------------------------

def test_good_calib_passes():
    """A synthetically correct calibration must pass with no warnings."""
    calib = _make_good_calib(rpe_px=0.3)
    ev = evaluate_calibration(calib)

    assert ev.passed, f"expected passed=True; warnings: {ev.warnings}"
    assert ev.warnings == []


def test_good_calib_epipolar_near_zero():
    """Epipolar error for a correctly rectified rig must be < 0.01 px."""
    calib = _make_good_calib()
    ev = evaluate_calibration(calib)

    assert ev.epipolar_error_mean_px < 0.01, (
        f"epipolar_error_mean_px {ev.epipolar_error_mean_px:.6f} px not near zero"
    )
    assert ev.epipolar_error_max_px < 0.05, (
        f"epipolar_error_max_px {ev.epipolar_error_max_px:.6f} px not near zero"
    )


def test_good_calib_symmetric_focal_length():
    """Symmetric calibration (K1 == K2) must report zero focal asymmetry."""
    calib = _make_good_calib()
    ev = evaluate_calibration(calib)

    assert ev.focal_length_asymmetry == pytest.approx(0.0, abs=1e-9)


def test_baseline_m_matches_calib():
    """baseline_m in the evaluation must match CalibrationResult.baseline_m."""
    calib = _make_good_calib(baseline=0.12)
    ev = evaluate_calibration(calib)

    assert ev.baseline_m == pytest.approx(calib.baseline_m, rel=1e-6)


def test_rpe_px_matches_calib():
    """rpe_px in the evaluation must echo CalibrationResult.rpe_px exactly."""
    calib = _make_good_calib(rpe_px=0.25)
    ev = evaluate_calibration(calib)

    assert ev.rpe_px == pytest.approx(0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests — threshold violations set passed=False and populate warnings
# ---------------------------------------------------------------------------

def test_high_rpe_fails():
    """rpe_px above max_rpe_px must set passed=False and add a warning."""
    calib = _make_good_calib(rpe_px=1.5)
    ev = evaluate_calibration(calib, max_rpe_px=0.5)

    assert not ev.passed
    assert any("rpe_px" in w for w in ev.warnings), (
        f"expected rpe_px warning; got: {ev.warnings}"
    )


def test_high_rpe_warning_mentions_value():
    """The rpe warning message must contain the actual rpe_px value."""
    calib = _make_good_calib(rpe_px=0.99)
    ev = evaluate_calibration(calib, max_rpe_px=0.5)

    rpe_warns = [w for w in ev.warnings if "rpe_px" in w]
    assert rpe_warns, "no rpe_px warning found"
    assert "0.990" in rpe_warns[0], (
        f"warning does not contain the rpe value: {rpe_warns[0]}"
    )


def test_asymmetric_focal_length_fails():
    """K2 with a different fx must produce a focal-asymmetry warning."""
    calib = _make_good_calib()
    K2_asym = calib.K2.copy()
    K2_asym[0, 0] *= 1.10   # 10 % difference
    calib = CalibrationResult(
        image_size=calib.image_size,
        K1=calib.K1, D1=calib.D1,
        K2=K2_asym,  D2=calib.D2,
        R=calib.R,   T=calib.T,
        baseline_m=calib.baseline_m,
        R1=calib.R1, R2=calib.R2, P1=calib.P1, P2=calib.P2, Q=calib.Q,
        rpe_px=0.0,
    )

    ev = evaluate_calibration(calib, max_focal_asymmetry=0.02)

    assert not ev.passed
    assert any("focal" in w for w in ev.warnings), (
        f"expected focal-length warning; got: {ev.warnings}"
    )


def test_bad_rectification_fails_epipolar():
    """Identity R1/R2 with a rotated rig must produce an epipolar warning.

    The 5-degree Y-axis rotation gives a mean epipolar error of ~0.7 px;
    tightening the threshold to 0.5 px makes the check fail as expected.
    """
    calib = _make_bad_rectification_calib(rot_deg=5.0)
    ev = evaluate_calibration(calib, max_epipolar_error_px=0.5)

    assert not ev.passed
    assert any("epipolar" in w for w in ev.warnings), (
        f"expected epipolar warning; got: {ev.warnings}"
    )


def test_bad_rectification_epipolar_error_large():
    """Epipolar error for an incorrectly rectified rig must be >> good-calibration error.

    Good calibration achieves < 0.01 px; identity R1/R2 with 5-degree rotation
    gives > 0.5 px mean error — two orders of magnitude worse.
    """
    calib = _make_bad_rectification_calib(rot_deg=5.0)
    ev = evaluate_calibration(calib)

    assert ev.epipolar_error_mean_px > 0.5, (
        f"expected large epipolar error, got {ev.epipolar_error_mean_px:.3f} px"
    )


def test_non_positive_baseline_fails():
    """baseline_m <= 0 must set passed=False regardless of other metrics."""
    calib = _make_good_calib(baseline=0.06)
    # Construct a copy with a broken baseline
    bad_calib = CalibrationResult(
        image_size=calib.image_size,
        K1=calib.K1, D1=calib.D1, K2=calib.K2, D2=calib.D2,
        R=calib.R,   T=calib.T,
        baseline_m=-0.01,
        R1=calib.R1, R2=calib.R2, P1=calib.P1, P2=calib.P2, Q=calib.Q,
        rpe_px=0.0,
    )

    ev = evaluate_calibration(bad_calib)

    assert not ev.passed
    assert any("baseline" in w for w in ev.warnings)


# ---------------------------------------------------------------------------
# Tests — multiple simultaneous failures
# ---------------------------------------------------------------------------

def test_multiple_failures_all_reported():
    """Both rpe and focal-asymmetry violations must each produce a warning."""
    calib = _make_good_calib(rpe_px=2.0)
    K2_asym = calib.K2.copy()
    K2_asym[0, 0] *= 1.10
    calib = CalibrationResult(
        image_size=calib.image_size,
        K1=calib.K1, D1=calib.D1,
        K2=K2_asym,  D2=calib.D2,
        R=calib.R,   T=calib.T,
        baseline_m=calib.baseline_m,
        R1=calib.R1, R2=calib.R2, P1=calib.P1, P2=calib.P2, Q=calib.Q,
        rpe_px=2.0,
    )

    ev = evaluate_calibration(calib, max_rpe_px=0.5, max_focal_asymmetry=0.02)

    assert not ev.passed
    assert len(ev.warnings) >= 2, (
        f"expected at least 2 warnings, got {len(ev.warnings)}: {ev.warnings}"
    )


# ---------------------------------------------------------------------------
# Tests — custom thresholds and n_test_points
# ---------------------------------------------------------------------------

def test_custom_threshold_raises_bar():
    """Setting a very tight max_rpe_px = 0.1 must fail a 0.3 px calibration."""
    calib = _make_good_calib(rpe_px=0.3)
    ev = evaluate_calibration(calib, max_rpe_px=0.1)

    assert not ev.passed


def test_custom_threshold_relaxed():
    """Setting a very loose max_rpe_px = 5.0 must pass a 0.3 px calibration."""
    calib = _make_good_calib(rpe_px=0.3)
    ev = evaluate_calibration(calib, max_rpe_px=5.0)

    # Only checking rpe — still need other metrics to pass
    rpe_warns = [w for w in ev.warnings if "rpe_px" in w]
    assert rpe_warns == [], "unexpected rpe warning with relaxed threshold"


def test_n_test_points_one():
    """n_test_points=1 must not crash and must return finite values."""
    calib = _make_good_calib()
    ev = evaluate_calibration(calib, n_test_points=1)

    assert np.isfinite(ev.epipolar_error_mean_px)
    assert np.isfinite(ev.epipolar_error_max_px)


def test_n_test_points_large():
    """n_test_points=200 must give consistent epipolar error for a good calib."""
    calib = _make_good_calib()
    ev = evaluate_calibration(calib, n_test_points=200)

    assert ev.epipolar_error_mean_px < 0.01
