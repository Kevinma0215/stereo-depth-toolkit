"""Tests for StereoPipeline.process_stack() and last_left_rect().

All tests are hardware-independent (synthetic data only).
"""
from __future__ import annotations

import numpy as np
import pytest

from stereo_depth.entities import CalibrationResult, DepthMap, FramePair, RectifiedPair
from stereo_depth.use_cases.pipeline import StereoPipeline
from stereo_depth.use_cases.ports import (
    IDepthEstimator,
    IDisparityMatcher,
    IRectifier,
)


# ---------------------------------------------------------------------------
# Synthetic calibration (image_size=(640, 480), baseline=0.06 m)
# ---------------------------------------------------------------------------

def _make_calib(image_size: tuple[int, int] = (640, 480)) -> CalibrationResult:
    """CalibrationResult with realistic HBVCAM-W202011HD values.

    fx=fy=480, cx=image_size[0]/2, cy=image_size[1]/2, baseline=0.06 m.
    """
    w, h = image_size
    fx = fy = 480.0
    cx, cy = w / 2.0, h / 2.0
    baseline = 0.06

    K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K2 = K1.copy()
    D1 = np.zeros(5, dtype=np.float64)
    D2 = np.zeros(5, dtype=np.float64)
    R  = np.eye(3, dtype=np.float64)
    T  = np.array([-baseline, 0.0, 0.0], dtype=np.float64)
    R1 = np.eye(3, dtype=np.float64)
    R2 = np.eye(3, dtype=np.float64)
    P1 = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
    P2 = np.array(
        [[fx, 0, cx, -fx * baseline], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64
    )
    Q = np.array(
        [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, fx],
            [0, 0, 1 / baseline, 0],
        ],
        dtype=np.float64,
    )
    return CalibrationResult(
        image_size=image_size,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T,
        baseline_m=baseline,
        R1=R1, R2=R2, P1=P1, P2=P2,
        Q=Q,
        rpe_px=0.0,
    )


# ---------------------------------------------------------------------------
# Lightweight port stubs
# ---------------------------------------------------------------------------

class _PassThroughRectifier(IRectifier):
    """Returns pair images unchanged as the RectifiedPair."""

    def rectify(self, pair: FramePair, calib: CalibrationResult) -> RectifiedPair:
        return RectifiedPair(left=pair.left, right=pair.right)


class _NullMatcher(IDisparityMatcher):
    """Returns an all-ones disparity map; intended for use with _ScriptedDepthEstimator."""

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        h, w = left.shape[:2]
        return np.ones((h, w), dtype=np.float32)


class _ScriptedDepthEstimator(IDepthEstimator):
    """Yields caller-supplied depth maps one per to_depth() call, ignoring disparity.

    This lets tests inject exact float32 depth arrays into the pipeline so that
    the spatial/temporal filtering logic can be verified in full isolation.
    """

    def __init__(self, depth_frames: list[np.ndarray]) -> None:
        self._remaining = list(depth_frames)

    def to_depth(self, disparity: np.ndarray, calib: CalibrationResult) -> DepthMap:
        data = self._remaining.pop(0).copy()
        return DepthMap(data=data, disparity=disparity)


class _FlatDisparityMatcher(IDisparityMatcher):
    """Returns a constant disparity map (default 20 px → 1.44 m at fx=480, b=0.06)."""

    def __init__(self, disparity: float = 20.0, hole_fraction: float = 0.0) -> None:
        self._disparity = disparity
        self._hole_fraction = hole_fraction

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        h, w = left.shape[:2]
        disp = np.full((h, w), self._disparity, dtype=np.float32)
        if self._hole_fraction > 0.0:
            rng = np.random.default_rng(0)
            mask = rng.random((h, w)) < self._hole_fraction
            disp[mask] = 0.0
        return disp


class _SimpleDepthEstimator(IDepthEstimator):
    """Computes depth via Z = fx * baseline / disparity; 0-disparity → NaN."""

    def to_depth(self, disparity: np.ndarray, calib: CalibrationResult) -> DepthMap:
        fx = float(calib.K1[0, 0])
        baseline = calib.baseline_m
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.where(
                disparity > 0,
                (fx * baseline / disparity).astype(np.float32),
                np.nan,
            ).astype(np.float32)
        return DepthMap(data=data, disparity=disparity)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(
    calib: CalibrationResult,
    disparity: float = 20.0,
    hole_fraction: float = 0.0,
) -> StereoPipeline:
    return StereoPipeline(
        rectifier=_PassThroughRectifier(),
        matcher=_FlatDisparityMatcher(disparity=disparity, hole_fraction=hole_fraction),
        depth_estimator=_SimpleDepthEstimator(),
        calib=calib,
    )


def _make_scripted_pipeline(
    calib: CalibrationResult,
    depth_frames: list[np.ndarray],
) -> StereoPipeline:
    """Pipeline that yields caller-supplied depth maps, bypassing disparity math."""
    return StereoPipeline(
        rectifier=_PassThroughRectifier(),
        matcher=_NullMatcher(),
        depth_estimator=_ScriptedDepthEstimator(depth_frames),
        calib=calib,
    )


def _make_sbs_frames(
    n: int,
    calib: CalibrationResult,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return *n* random uint8 BGR SBS frames matching calib.image_size."""
    w, h = calib.image_size
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, (h, w * 2, 3), dtype=np.uint8)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests — output shape and dtype
# ---------------------------------------------------------------------------

def test_output_shape_and_dtype():
    """process_stack() must return a float32 array of shape (H, W)."""
    calib = _make_calib()
    w, h = calib.image_size
    pipeline = _make_pipeline(calib)
    frames = _make_sbs_frames(15, calib)

    result = pipeline.process_stack(frames)

    assert result.shape == (h, w), f"expected ({h}, {w}), got {result.shape}"
    assert result.dtype == np.float32, f"expected float32, got {result.dtype}"


def test_output_shape_accepts_ndarray_input():
    """process_stack() must also accept a numpy array (not just a list)."""
    calib = _make_calib()
    w, h = calib.image_size
    pipeline = _make_pipeline(calib)
    frames = np.stack(_make_sbs_frames(15, calib), axis=0)  # (N, H, W*2, 3)

    result = pipeline.process_stack(frames)

    assert result.shape == (h, w)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests — NaN masking
# ---------------------------------------------------------------------------

def test_valid_depths_not_nan():
    """Pixels with depth >= min_depth_m must not be NaN."""
    calib = _make_calib()
    pipeline = _make_pipeline(calib, disparity=20.0)  # → ~1.44 m
    frames = _make_sbs_frames(15, calib)

    result = pipeline.process_stack(frames, min_depth_m=0.10)

    valid = result[~np.isnan(result)]
    assert valid.size > 0, "all pixels are NaN — expected valid depth values"


def test_below_min_depth_masked_as_nan():
    """Pixels with depth < min_depth_m must be NaN in the output."""
    calib = _make_calib()
    # disparity=20 → Z ≈ 1.44 m; set min_depth_m above that to force all NaN
    pipeline = _make_pipeline(calib, disparity=20.0)
    frames = _make_sbs_frames(15, calib)

    result = pipeline.process_stack(frames, min_depth_m=2.0)

    assert np.all(np.isnan(result)), (
        "expected all pixels to be NaN when min_depth_m exceeds all depths"
    )


def test_zero_disparity_pixels_are_nan():
    """Pixels where the matcher returns 0 (no match) must end up as NaN."""
    calib = _make_calib()
    # All disparity pixels set to 0 → all depths are NaN/invalid
    pipeline = _make_pipeline(calib, disparity=0.0)
    frames = _make_sbs_frames(15, calib)

    result = pipeline.process_stack(frames, min_depth_m=0.10)

    assert np.all(np.isnan(result)), (
        "expected all NaN when matcher always returns 0 disparity"
    )


# ---------------------------------------------------------------------------
# Tests — ValueError on bad inputs
# ---------------------------------------------------------------------------

def test_wrong_frame_count_raises():
    """len(frames) != temporal_frames must raise ValueError."""
    calib = _make_calib()
    pipeline = _make_pipeline(calib)
    frames = _make_sbs_frames(10, calib)  # 10 frames but temporal_frames=15

    with pytest.raises(ValueError, match="15"):
        pipeline.process_stack(frames, temporal_frames=15)


def test_even_kernel_raises():
    """Even spatial_kernel must raise ValueError."""
    calib = _make_calib()
    pipeline = _make_pipeline(calib)
    frames = _make_sbs_frames(5, calib)

    with pytest.raises(ValueError, match="odd"):
        pipeline.process_stack(frames, spatial_kernel=4, temporal_frames=5)


def test_kernel_less_than_3_raises():
    """spatial_kernel < 3 must raise ValueError."""
    calib = _make_calib()
    pipeline = _make_pipeline(calib)
    frames = _make_sbs_frames(5, calib)

    with pytest.raises(ValueError, match="odd"):
        pipeline.process_stack(frames, spatial_kernel=1, temporal_frames=5)


def test_wrong_frame_shape_raises():
    """Frames with an unexpected shape must raise ValueError."""
    calib = _make_calib()
    pipeline = _make_pipeline(calib)
    # Use the wrong width (100 instead of calib.image_size[0] * 2)
    bad_frames = [np.zeros((480, 100, 3), dtype=np.uint8)] * 5

    with pytest.raises(ValueError, match="shape"):
        pipeline.process_stack(bad_frames, temporal_frames=5)


# ---------------------------------------------------------------------------
# Tests — last_left_rect()
# ---------------------------------------------------------------------------

def test_last_left_rect_before_call_raises():
    """last_left_rect() before process_stack() must raise RuntimeError."""
    calib = _make_calib()
    pipeline = _make_pipeline(calib)

    with pytest.raises(RuntimeError):
        pipeline.last_left_rect()


def test_last_left_rect_shape_and_dtype():
    """last_left_rect() must return a (H, W, 3) uint8 BGR image."""
    calib = _make_calib()
    w, h = calib.image_size
    pipeline = _make_pipeline(calib)
    frames = _make_sbs_frames(15, calib)

    pipeline.process_stack(frames)
    left_rect = pipeline.last_left_rect()

    assert left_rect.shape == (h, w, 3), (
        f"expected ({h}, {w}, 3), got {left_rect.shape}"
    )
    assert left_rect.dtype == np.uint8, f"expected uint8, got {left_rect.dtype}"


# ---------------------------------------------------------------------------
# Level 1 — output contract (strict)
# ---------------------------------------------------------------------------

def test_output_is_strictly_2d():
    """Output must be exactly 2-D (H, W), not (H, W, 1) or similar."""
    calib = _make_calib()
    pipeline = _make_pipeline(calib)
    frames = _make_sbs_frames(15, calib)

    result = pipeline.process_stack(frames)

    assert result.ndim == 2, f"expected ndim=2, got ndim={result.ndim}"


def test_masked_pixels_are_nan_not_zero():
    """Pixels that fall below min_depth_m must be NaN, never 0.0."""
    calib = _make_calib()
    # disparity=20 → Z ≈ 1.44 m; raise min_depth_m above that to mask everything
    pipeline = _make_pipeline(calib, disparity=20.0)
    frames = _make_sbs_frames(15, calib)

    result = pipeline.process_stack(frames, min_depth_m=2.0)

    assert not np.any(result == 0.0), (
        "masked pixels must be NaN, not 0.0"
    )
    assert np.all(np.isnan(result)), (
        "expected every pixel to be NaN when min_depth_m exceeds all depths"
    )


def test_valid_pixels_are_not_nan():
    """Every pixel whose depth exceeds min_depth_m must be finite, not NaN."""
    calib = _make_calib()
    w, h = calib.image_size
    # Inject a uniform 1.0 m depth map for every frame
    depth_frames = [np.full((h, w), 1.0, dtype=np.float32) for _ in range(15)]
    pipeline = _make_scripted_pipeline(calib, depth_frames)
    frames = _make_sbs_frames(15, calib)

    result = pipeline.process_stack(frames, min_depth_m=0.10)

    assert not np.any(np.isnan(result)), (
        "no pixel should be NaN when all depths are well above min_depth_m"
    )


# ---------------------------------------------------------------------------
# Level 1 — temporal median correctness
# ---------------------------------------------------------------------------

def test_temporal_median_odd_n():
    """Pixel-wise temporal median is correct for an odd number of frames.

    Each frame is flat at a distinct depth value.  A flat frame is unchanged
    by the spatial filter, so the output equals the median of the per-frame
    values — which can be verified by hand.
    """
    calib = _make_calib()
    w, h = calib.image_size
    n = 5
    # Inject values in a non-sorted order so we're really testing median, not min/max.
    values = [1.5, 3.5, 2.5, 5.5, 4.5]  # sorted: 1.5 2.5 3.5 4.5 5.5 → median = 3.5
    expected = float(np.median(values))  # 3.5

    depth_frames = [np.full((h, w), v, dtype=np.float32) for v in values]
    pipeline = _make_scripted_pipeline(calib, depth_frames)
    frames = _make_sbs_frames(n, calib)

    result = pipeline.process_stack(frames, spatial_kernel=3, temporal_frames=n, min_depth_m=0.1)

    cy, cx = h // 2, w // 2
    assert result[cy, cx] == pytest.approx(expected, abs=1e-5), (
        f"temporal median at ({cy}, {cx}): expected {expected}, got {result[cy, cx]}"
    )


def test_temporal_median_even_n():
    """Pixel-wise temporal median is correct for an even number of frames.

    For even N, numpy defines the median as the mean of the two middle values.
    """
    calib = _make_calib()
    w, h = calib.image_size
    n = 4
    values = [1.5, 3.5, 2.5, 5.5]  # sorted: 1.5 2.5 3.5 5.5 → median = (2.5+3.5)/2 = 3.0
    expected = float(np.median(values))  # 3.0

    depth_frames = [np.full((h, w), v, dtype=np.float32) for v in values]
    pipeline = _make_scripted_pipeline(calib, depth_frames)
    frames = _make_sbs_frames(n, calib)

    result = pipeline.process_stack(frames, spatial_kernel=3, temporal_frames=n, min_depth_m=0.1)

    cy, cx = h // 2, w // 2
    assert result[cy, cx] == pytest.approx(expected, abs=1e-5), (
        f"temporal median at ({cy}, {cx}): expected {expected}, got {result[cy, cx]}"
    )


# ---------------------------------------------------------------------------
# Level 1 — spatial median suppresses a single spike
# ---------------------------------------------------------------------------

def test_spatial_median_suppresses_spike():
    """A lone bright spike must be reduced to the surrounding value after medianBlur.

    A single pixel set to 99.0 m is surrounded by 1.0 m in all directions.
    In the 5x5 window centred on the spike, 24 of 25 values are 1.0 and
    1 is 99.0; the median of those 25 values is 1.0.
    """
    calib = _make_calib()
    w, h = calib.image_size
    ry, cx = h // 2, w // 2   # center pixel — well away from any border

    flat = np.full((h, w), 1.0, dtype=np.float32)
    spiked = flat.copy()
    spiked[ry, cx] = 99.0

    # Single frame so only spatial filtering is exercised
    pipeline = _make_scripted_pipeline(calib, [spiked])
    frames = _make_sbs_frames(1, calib)

    result = pipeline.process_stack(
        frames, spatial_kernel=5, temporal_frames=1, min_depth_m=0.1
    )

    assert result[ry, cx] == pytest.approx(1.0, abs=0.1), (
        f"spike at ({ry}, {cx}) not suppressed: got {result[ry, cx]:.4f} m, expected ≈1.0 m"
    )


# ---------------------------------------------------------------------------
# Level 1 — side effects
# ---------------------------------------------------------------------------

def test_input_frames_not_mutated():
    """process_stack() must not alter the caller's frame list or its arrays."""
    calib = _make_calib()
    pipeline = _make_pipeline(calib)
    frames = _make_sbs_frames(15, calib)
    originals = [f.copy() for f in frames]

    pipeline.process_stack(frames)

    for i, (original, current) in enumerate(zip(originals, frames)):
        assert np.array_equal(original, current), (
            f"frame {i} was mutated by process_stack()"
        )


def test_process_stack_idempotent():
    """Calling process_stack() twice with the same input must return identical arrays."""
    calib = _make_calib()
    pipeline = _make_pipeline(calib, disparity=20.0)
    frames = _make_sbs_frames(15, calib)

    result1 = pipeline.process_stack(frames)
    result2 = pipeline.process_stack(frames)

    # np.testing.assert_array_equal treats NaN == NaN as equal
    np.testing.assert_array_equal(result1, result2)
