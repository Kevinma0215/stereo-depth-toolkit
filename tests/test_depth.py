"""Tests for SgbmMatcher.

All tests are hardware-independent: stereo pairs are synthesised by
horizontally shifting a random texture image.
"""
from __future__ import annotations
import numpy as np
import cv2
import pytest

from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stereo_pair(
    shift: int = 20, h: int = 240, w: int = 320, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a rectified pair by shifting a random texture image.

    The right image is the left image shifted `shift` pixels to the left,
    which simulates a positive disparity of `shift` pixels for every pixel
    that has a valid match.
    """
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (h, w), dtype=np.uint8)
    # Shift right image left by `shift` pixels (pad right edge with 0)
    right = np.zeros_like(left)
    right[:, : w - shift] = left[:, shift:]
    return left, right


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset_name", ["indoor", "outdoor", "high_quality"])
def test_output_shape_and_dtype(preset_name: str):
    """compute() must return float32 disparity with the same H×W as input."""
    left, right = _make_stereo_pair()
    h, w = left.shape

    matcher = SgbmMatcher(preset_name=preset_name)
    disp = matcher.compute(left, right)

    assert disp.shape == (h, w), f"expected shape ({h}, {w}), got {disp.shape}"
    assert disp.dtype == np.float32, f"expected float32, got {disp.dtype}"


def test_bgr_input_accepted():
    """compute() must accept 3-channel BGR images as well as grayscale."""
    left_gray, right_gray = _make_stereo_pair()
    left_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)

    matcher = SgbmMatcher()
    disp_gray = matcher.compute(left_gray, right_gray)
    disp_bgr = matcher.compute(left_bgr, right_bgr)

    assert disp_gray.shape == disp_bgr.shape
    assert disp_bgr.dtype == np.float32


def test_disparity_values_match_shift():
    """Pixels with a valid match must report disparity close to the known shift."""
    shift = 20
    left, right = _make_stereo_pair(shift=shift)
    matcher = SgbmMatcher(preset_name="indoor")
    disp = matcher.compute(left, right)

    valid = disp[disp > 0]
    assert valid.size > 0, "no positive disparity found at all"
    # All matched pixels should agree with the known horizontal shift (±1 px)
    assert np.median(valid) == pytest.approx(shift, abs=1.0), (
        f"median disparity {np.median(valid):.2f} px not close to shift {shift} px"
    )


def test_invalid_preset_raises():
    """Passing an unknown preset name must raise ValueError immediately."""
    with pytest.raises(ValueError, match="Unknown preset"):
        SgbmMatcher(preset_name="nonexistent")
