"""Tests for CudaBmMatcher.

All tests are hardware-independent in the sense that they use synthetic
stereo pairs.  The entire suite is auto-skipped when OpenCV CUDA support
is absent or no CUDA-capable GPU is present — matching the skip pattern
used by test_retinify.py.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level skip when CUDA is unavailable
# ---------------------------------------------------------------------------

try:
    from stereo_depth.adapters.matcher.cuda_bm_matcher import CudaBmMatcher
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="OpenCV CUDA support not available (cv2.cuda.StereoBM_create missing or no GPU)",
)


# ---------------------------------------------------------------------------
# Helpers (mirror test_depth.py)
# ---------------------------------------------------------------------------

def _make_stereo_pair(
    shift: int = 20, h: int = 240, w: int = 320, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a rectified pair by shifting a random texture image.

    The right image is the left image shifted ``shift`` pixels to the left,
    producing a positive disparity of ``shift`` pixels for every matched pixel.
    """
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (h, w), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, : w - shift] = left[:, shift:]
    return left, right


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape_and_dtype():
    """compute() must return float32 disparity with the same H×W as input."""
    left, right = _make_stereo_pair(h=480, w=640)
    matcher = CudaBmMatcher(num_disparities=64, block_size=15)
    disp = matcher.compute(left, right)

    assert disp.shape == (480, 640), f"expected (480, 640), got {disp.shape}"
    assert disp.dtype == np.float32, f"expected float32, got {disp.dtype}"


def test_bgr_input_accepted():
    """compute() must accept 3-channel BGR images as well as grayscale."""
    import cv2
    left_gray, right_gray = _make_stereo_pair()
    left_bgr  = cv2.cvtColor(left_gray,  cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)

    matcher = CudaBmMatcher()
    disp_gray = matcher.compute(left_gray, right_gray)
    disp_bgr  = matcher.compute(left_bgr,  right_bgr)

    assert disp_gray.shape == disp_bgr.shape
    assert disp_bgr.dtype == np.float32


def test_disparity_values_match_shift():
    """Valid matched pixels must report disparity close to the known shift.

    Verifies depth error < 5 % at 0.5–1.5 m on a synthetic known-distance
    target (equivalent to the architecture doc's disparity sanity test spec).
    Uses focal_length=480 px, baseline=0.06 m — the HBVCAM-W202011HD values.
    """
    fx, baseline = 480.0, 0.06
    # depth = fx * baseline / disparity  =>  shift chosen so depth ≈ 1.0 m
    shift = int(round(fx * baseline / 1.0))   # ≈ 29 px
    left, right = _make_stereo_pair(shift=shift, h=240, w=320)

    matcher = CudaBmMatcher(num_disparities=64, block_size=15, prefilter_cap=31)
    disp = matcher.compute(left, right)

    valid = disp[disp > 0]
    assert valid.size > 0, "No positive disparity found in output"

    median_disp = float(np.median(valid))
    assert median_disp == pytest.approx(shift, abs=1.5), (
        f"Median disparity {median_disp:.2f} px is not within 1.5 px of "
        f"expected shift {shift} px"
    )

    # Convert to depth and verify < 5 % error at 1.0 m
    depth_m = fx * baseline / median_disp
    assert abs(depth_m - 1.0) / 1.0 < 0.05, (
        f"Depth error {abs(depth_m - 1.0) / 1.0 * 100:.1f}% exceeds 5% threshold"
    )


def test_invalid_num_disparities_raises():
    """num_disparities not divisible by 16 must raise ValueError immediately."""
    with pytest.raises(ValueError, match="divisible by 16"):
        CudaBmMatcher(num_disparities=50)


def test_invalid_block_size_raises():
    """block_size outside [5, 31] or even must raise ValueError immediately."""
    with pytest.raises(ValueError, match="odd"):
        CudaBmMatcher(block_size=4)   # too small and even
    with pytest.raises(ValueError, match="odd"):
        CudaBmMatcher(block_size=33)  # too large
    with pytest.raises(ValueError, match="odd"):
        CudaBmMatcher(block_size=16)  # even


def test_multiple_frames_independent():
    """Successive compute() calls must not corrupt each other's output."""
    left, right = _make_stereo_pair(shift=20)
    matcher = CudaBmMatcher()

    disp1 = matcher.compute(left, right)
    disp2 = matcher.compute(left, right)

    np.testing.assert_array_equal(disp1, disp2)
