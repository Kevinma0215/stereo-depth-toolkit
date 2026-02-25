"""Tests for ICameraSource.stream() implementations.

All tests are hardware-independent: images are synthesised in a tmp_path
fixture, so no camera or external files are required.
"""
from __future__ import annotations
import numpy as np
import cv2
import pytest

from stereo_depth.adapters.camera.file_source import FileSource
from stereo_depth.entities import FramePair

H, W = 60, 80
N_FRAMES = 3


@pytest.fixture()
def frame_dir(tmp_path):
    """Create N_FRAMES synthetic BGR image pairs under tmp_path/left/ and tmp_path/right/."""
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(N_FRAMES):
        img_l = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        img_r = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        cv2.imwrite(str(left_dir / f"{i:04d}.png"), img_l)
        cv2.imwrite(str(right_dir / f"{i:04d}.png"), img_r)
    return tmp_path


def test_file_source_stream_yields_correct_count(frame_dir):
    source = FileSource(frame_dir / "left", frame_dir / "right")
    frames = list(source.stream())
    assert len(frames) == N_FRAMES


def test_file_source_stream_yields_frame_pairs(frame_dir):
    source = FileSource(frame_dir / "left", frame_dir / "right")
    for fp in source.stream():
        assert isinstance(fp, FramePair)


def test_file_source_stream_shape(frame_dir):
    source = FileSource(frame_dir / "left", frame_dir / "right")
    for fp in source.stream():
        assert fp.left.shape == (H, W, 3)
        assert fp.right.shape == (H, W, 3)
        assert fp.left.dtype == np.uint8
        assert fp.right.dtype == np.uint8
