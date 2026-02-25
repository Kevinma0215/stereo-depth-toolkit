"""Tests for RetinifyMatcher and retinify_adapter.

RetinifyMatcher tests require the ``retinify`` package (CUDA/TensorRT GPU
library).  They are automatically skipped when it is not installed.

The retinify_adapter tests have no such dependency and always run.
"""
from __future__ import annotations
import importlib.util
import json
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Detect retinify availability without triggering the module-level ImportError
# in retinify_matcher.py.  We check the underlying package first.
# ---------------------------------------------------------------------------
_HAS_RETINIFY = importlib.util.find_spec("retinify") is not None

if _HAS_RETINIFY:
    from stereo_depth.adapters.matcher.retinify_matcher import RetinifyMatcher

skip_no_retinify = pytest.mark.skipif(
    not _HAS_RETINIFY,
    reason="retinify not installed (requires CUDA/TensorRT GPU)",
)

# ---------------------------------------------------------------------------
# Imports that are always safe
# ---------------------------------------------------------------------------
from stereo_depth.adapters.calibration.retinify_adapter import (
    calibration_result_to_retinify,
)
from stereo_depth.entities import CalibrationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calib() -> CalibrationResult:
    f, w, h, b = 500.0, 640, 480, 0.06
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    I3 = np.eye(3, dtype=np.float64)
    P1 = np.hstack([K, np.zeros((3, 1))])
    T = np.array([-b, 0.0, 0.0], dtype=np.float64)
    P2 = np.hstack([K, np.array([[K[0, 0] * -b], [0], [0]])])
    Q = np.eye(4, dtype=np.float64)
    return CalibrationResult(
        image_size=(w, h),
        K1=K, D1=D, K2=K.copy(), D2=D.copy(),
        R=I3, T=T, baseline_m=b,
        R1=I3, R2=I3, P1=P1, P2=P2, Q=Q,
        rpe_px=0.0,
    )


# ---------------------------------------------------------------------------
# retinify_adapter tests  (always run)
# ---------------------------------------------------------------------------

class TestCalibrationResultToRetinify:
    def test_required_keys_present(self):
        d = calibration_result_to_retinify(_make_calib())
        for key in ("image_size", "K1", "D1", "K2", "D2", "R", "T", "baseline_m"):
            assert key in d, f"missing key: {key}"

    def test_image_size_values(self):
        calib = _make_calib()
        d = calibration_result_to_retinify(calib)
        assert d["image_size"]["width"] == calib.image_size[0]
        assert d["image_size"]["height"] == calib.image_size[1]

    def test_baseline_m_value(self):
        calib = _make_calib()
        d = calibration_result_to_retinify(calib)
        assert d["baseline_m"] == pytest.approx(calib.baseline_m)

    def test_matrix_shapes(self):
        d = calibration_result_to_retinify(_make_calib())
        assert len(d["K1"]) == 3 and len(d["K1"][0]) == 3
        assert len(d["R"]) == 3 and len(d["R"][0]) == 3
        assert len(d["T"]) == 3

    def test_output_is_json_serialisable(self):
        d = calibration_result_to_retinify(_make_calib())
        serialised = json.dumps(d)   # must not raise
        assert len(serialised) > 0

    def test_rectification_matrices_excluded(self):
        """R1/R2/P1/P2/Q are NOT included; retinify computes them itself."""
        d = calibration_result_to_retinify(_make_calib())
        for key in ("R1", "R2", "P1", "P2", "Q"):
            assert key not in d, f"unexpected key in retinify dict: {key}"


# ---------------------------------------------------------------------------
# RetinifyMatcher tests  (skipped when retinify not installed)
# ---------------------------------------------------------------------------

@skip_no_retinify
def test_retinify_matcher_import_error_message():
    """When retinify IS available, importing the module must succeed."""
    # If we get here, _HAS_RETINIFY is True and the import already succeeded.
    assert RetinifyMatcher is not None


@skip_no_retinify
def test_retinify_matcher_invalid_mode():
    with pytest.raises(ValueError, match="Unknown Retinify mode"):
        RetinifyMatcher(width=640, height=480, mode="invalid_mode")


@skip_no_retinify
@pytest.mark.parametrize("mode", ["fast", "balanced", "accurate"])
def test_retinify_matcher_output_shape(mode):
    h, w = 480, 640
    matcher = RetinifyMatcher(width=w, height=h, mode=mode)
    left = np.zeros((h, w, 3), dtype=np.uint8)
    right = np.zeros((h, w, 3), dtype=np.uint8)
    disp = matcher.compute(left, right)
    assert disp.shape == (h, w)
    assert disp.dtype == np.float32


# ---------------------------------------------------------------------------
# ImportError guard test  (always run — verifies the guard works without retinify)
# ---------------------------------------------------------------------------

def test_retinify_matcher_import_error_when_absent(monkeypatch):
    """Importing retinify_matcher when retinify is absent must raise ImportError
    with a helpful install message."""
    if _HAS_RETINIFY:
        pytest.skip("retinify is installed — cannot test absence guard")

    import importlib
    import sys

    # Ensure the module is not already cached
    mod_name = "stereo_depth.adapters.matcher.retinify_matcher"
    sys.modules.pop(mod_name, None)

    with pytest.raises(ImportError, match="pip install retinify"):
        importlib.import_module(mod_name)
