"""Tests for the IPostProcessor port and its adapters."""
from __future__ import annotations
import numpy as np
import pytest

from stereo_depth.entities import DepthMap, CalibrationResult
from stereo_depth.entities.frame import FramePair
from stereo_depth.use_cases.ports import IPostProcessor
from stereo_depth.adapters.post_processor.identity_post_processor import IdentityPostProcessor
from stereo_depth.adapters.post_processor.hole_fill_post_processor import HoleFillPostProcessor
from stereo_depth.adapters.depth.opencv_depth_estimator import OpenCVDepthEstimator
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.use_cases.pipeline import StereoPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_calib() -> CalibrationResult:
    fx = 480.0
    cx, cy = 320.0, 240.0
    Tx = -fx * 0.06
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0,  fx],
        [0, 0, 1 / 0.06, 0],
    ], dtype=np.float64)
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = fx
    K[0, 2] = cx; K[1, 2] = cy
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


def _make_depth_map(h: int = 60, w: int = 80, with_holes: bool = False) -> DepthMap:
    """Synthetic DepthMap (flat at 1.0 m, with optional hole region)."""
    data = np.ones((h, w), dtype=np.float32)
    disp = np.full((h, w), 28.8, dtype=np.float32)   # 480*0.06/1.0 = 28.8 px
    if with_holes:
        data[20:40, 30:50] = np.nan
        disp[20:40, 30:50] = 0.0
    return DepthMap(data=data, disparity=disp)


# ---------------------------------------------------------------------------
# IPostProcessor interface contract
# ---------------------------------------------------------------------------

class TestIPostProcessorContract:
    def test_identity_is_ipostprocessor(self):
        assert isinstance(IdentityPostProcessor(), IPostProcessor)

    def test_hole_fill_is_ipostprocessor(self):
        assert isinstance(HoleFillPostProcessor(), IPostProcessor)


# ---------------------------------------------------------------------------
# IdentityPostProcessor
# ---------------------------------------------------------------------------

class TestIdentityPostProcessor:
    def test_returns_same_object(self):
        pp = IdentityPostProcessor()
        dm = _make_depth_map()
        assert pp.process(dm) is dm

    def test_data_unchanged(self):
        pp = IdentityPostProcessor()
        dm = _make_depth_map(with_holes=True)
        out = pp.process(dm)
        np.testing.assert_array_equal(out.data, dm.data)

    def test_disparity_unchanged(self):
        pp = IdentityPostProcessor()
        dm = _make_depth_map()
        out = pp.process(dm)
        np.testing.assert_array_equal(out.disparity, dm.disparity)


# ---------------------------------------------------------------------------
# HoleFillPostProcessor
# ---------------------------------------------------------------------------

class TestHoleFillPostProcessor:
    def test_fill_reduces_nan_count(self):
        pp = HoleFillPostProcessor(radius=3)
        dm = _make_depth_map(with_holes=True)
        nan_before = int(np.sum(~np.isfinite(dm.data)))
        out = pp.process(dm)
        nan_after = int(np.sum(~np.isfinite(out.data)))
        assert nan_after < nan_before

    def test_valid_pixels_preserved_exactly(self):
        pp = HoleFillPostProcessor(radius=3)
        dm = _make_depth_map(with_holes=True)
        out = pp.process(dm)
        valid = np.isfinite(dm.data)
        np.testing.assert_array_equal(out.data[valid], dm.data[valid])

    def test_filled_depth_positive(self):
        pp = HoleFillPostProcessor(radius=3)
        dm = _make_depth_map(with_holes=True)
        out = pp.process(dm)
        finite = out.data[np.isfinite(out.data)]
        assert np.all(finite > 0)

    def test_disparity_unmodified(self):
        pp = HoleFillPostProcessor(radius=3)
        dm = _make_depth_map(with_holes=True)
        out = pp.process(dm)
        np.testing.assert_array_equal(out.disparity, dm.disparity)

    def test_left_right_rect_forwarded(self):
        pp = HoleFillPostProcessor()
        left  = np.zeros((60, 80, 3), dtype=np.uint8)
        right = np.zeros((60, 80, 3), dtype=np.uint8)
        dm = DepthMap(
            data=np.ones((60, 80), dtype=np.float32),
            disparity=np.ones((60, 80), dtype=np.float32),
            left_rect=left, right_rect=right,
        )
        out = pp.process(dm)
        assert out.left_rect is left
        assert out.right_rect is right

    def test_no_holes_returns_identical_data(self):
        pp = HoleFillPostProcessor(radius=3)
        dm = _make_depth_map(with_holes=False)
        out = pp.process(dm)
        np.testing.assert_array_equal(out.data, dm.data)

    def test_radius_zero_raises(self):
        with pytest.raises(ValueError, match="radius"):
            HoleFillPostProcessor(radius=0)

    def test_radius_negative_raises(self):
        with pytest.raises(ValueError, match="radius"):
            HoleFillPostProcessor(radius=-1)

    def test_larger_radius_fills_at_least_as_much(self):
        dm_small  = _make_depth_map(h=80, w=80, with_holes=True)
        dm_large  = _make_depth_map(h=80, w=80, with_holes=True)
        out_small = HoleFillPostProcessor(radius=1).process(dm_small)
        out_large = HoleFillPostProcessor(radius=10).process(dm_large)
        nan_small = int(np.sum(~np.isfinite(out_small.data)))
        nan_large = int(np.sum(~np.isfinite(out_large.data)))
        assert nan_large <= nan_small


# ---------------------------------------------------------------------------
# StereoPipeline integration
# ---------------------------------------------------------------------------

def _build_pipeline(
    calib: CalibrationResult,
    post_processors=None,
) -> StereoPipeline:
    return StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=SgbmMatcher(preset_name="indoor"),
        depth_estimator=OpenCVDepthEstimator(),
        calib=calib,
        post_processors=post_processors,
    )


def _make_frame_pair(calib: CalibrationResult, disparity_px: float = 24.0) -> FramePair:
    """Synthetic rectified-like stereo pair with a uniform horizontal shift."""
    h, w = calib.image_size[1], calib.image_size[0]
    rng  = np.random.default_rng(0)
    left = (rng.integers(30, 200, (h, w, 3), dtype=np.uint8))
    # Shift left image by disparity_px columns to create right image
    shift = int(disparity_px)
    right = np.zeros_like(left)
    if shift < w:
        right[:, :w - shift] = left[:, shift:]
    return FramePair(left=left, right=right)


class TestStereoPipelinePostProcessors:
    def test_no_post_processors_runs(self):
        calib = _make_calib()
        pipeline = _build_pipeline(calib)
        pair = _make_frame_pair(calib)
        dm = pipeline.process(pair)
        assert dm.data.shape == (calib.image_size[1], calib.image_size[0])

    def test_empty_list_same_as_none(self):
        calib = _make_calib()
        pp_none = _build_pipeline(calib, post_processors=None)
        pp_empty = _build_pipeline(calib, post_processors=[])
        pair = _make_frame_pair(calib)
        dm_none  = pp_none.process(pair)
        dm_empty = pp_empty.process(pair)
        np.testing.assert_array_equal(dm_none.data, dm_empty.data)

    def test_identity_post_processor_unchanged(self):
        calib = _make_calib()
        pp_base = _build_pipeline(calib)
        pp_id   = _build_pipeline(calib, post_processors=[IdentityPostProcessor()])
        pair = _make_frame_pair(calib)
        dm_base = pp_base.process(pair)
        dm_id   = pp_id.process(pair)
        np.testing.assert_array_equal(dm_base.data, dm_id.data)

    def test_hole_fill_post_processor_reduces_nans(self):
        calib = _make_calib()
        pp_base = _build_pipeline(calib)
        pp_fill = _build_pipeline(calib, post_processors=[HoleFillPostProcessor(radius=3)])
        pair = _make_frame_pair(calib)
        dm_base = pp_base.process(pair)
        dm_fill = pp_fill.process(pair)
        nan_base = int(np.sum(~np.isfinite(dm_base.data)))
        nan_fill = int(np.sum(~np.isfinite(dm_fill.data)))
        assert nan_fill <= nan_base

    def test_post_processors_applied_in_order(self):
        """A chain [IdentityPostProcessor, HoleFillPostProcessor] should fill holes."""
        calib = _make_calib()
        chain = [IdentityPostProcessor(), HoleFillPostProcessor(radius=3)]
        pp_chain = _build_pipeline(calib, post_processors=chain)
        pp_fill  = _build_pipeline(calib, post_processors=[HoleFillPostProcessor(radius=3)])
        pair = _make_frame_pair(calib)
        dm_chain = pp_chain.process(pair)
        dm_fill  = pp_fill.process(pair)
        np.testing.assert_array_equal(dm_chain.data, dm_fill.data)

    def test_left_right_rect_preserved_through_chain(self):
        calib = _make_calib()
        pp = _build_pipeline(calib, post_processors=[IdentityPostProcessor()])
        pair = _make_frame_pair(calib)
        dm = pp.process(pair)
        assert dm.left_rect  is not None
        assert dm.right_rect is not None
