"""Tests for CameraStreamer (hardware-independent via MockSource / MockProcessor)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from stereo_depth.entities import FramePair
from stereo_depth.rolling_buffer import RollingBuffer
from stereo_depth.camera_streamer import CameraStreamer, NotReadyError, SnapshotResult


# ---------------------------------------------------------------------------
# Mock collaborators
# ---------------------------------------------------------------------------

class MockSource:
    def grab(self) -> FramePair:
        left  = np.zeros((720, 1280, 3), dtype=np.uint8)
        right = np.zeros((720, 1280, 3), dtype=np.uint8)
        return FramePair(left=left, right=right)

    def release(self) -> None:
        pass


class MockProcessor:
    def process_stack(self, frames) -> np.ndarray:
        self._last = np.zeros((720, 1280), dtype=np.float32)
        return self._last

    @property
    def last_left_rect(self) -> np.ndarray:
        return np.zeros((720, 1280, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_streamer(maxlen: int = 15) -> CameraStreamer:
    return CameraStreamer(
        source=MockSource(),
        buffer=RollingBuffer(maxlen=maxlen),
        processor=MockProcessor(),
    )


def _wait_ready(streamer: CameraStreamer, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if streamer.is_ready:
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_not_ready_before_buffer_full():
    """snapshot() must raise NotReadyError before the buffer is full."""
    streamer = _make_streamer()
    streamer.start()
    try:
        with pytest.raises(NotReadyError):
            streamer.snapshot()
    finally:
        streamer.stop()


def test_is_ready_after_buffer_fills():
    """is_ready must become True once the buffer holds maxlen frames."""
    streamer = _make_streamer()
    streamer.start()
    try:
        assert _wait_ready(streamer), "Buffer did not fill within 5 s"
        assert streamer.is_ready is True
    finally:
        streamer.stop()


def test_snapshot_returns_correct_types():
    """snapshot() must return a SnapshotResult with correct shapes and types."""
    streamer = _make_streamer()
    streamer.start()
    try:
        assert _wait_ready(streamer), "Buffer did not fill within 5 s"
        result = streamer.snapshot()

        assert isinstance(result, SnapshotResult)
        assert result.rgb_snapshot.shape  == (720, 1280, 3)
        assert result.stable_depth.shape  == (720, 1280)
        assert result.rgb_snapshot.dtype  == np.uint8
        assert result.stable_depth.dtype  == np.float32
        assert result.process_time_s      >= 0.0
    finally:
        streamer.stop()


def test_snapshot_frame_index_increases():
    """frame_index in a later snapshot must be >= that of an earlier snapshot."""
    streamer = _make_streamer()
    streamer.start()
    try:
        assert _wait_ready(streamer), "Buffer did not fill within 5 s"
        first = streamer.snapshot()
        time.sleep(0.1)
        second = streamer.snapshot()

        assert second.frame_index >= first.frame_index
    finally:
        streamer.stop()


def test_stop_is_clean():
    """start() followed by stop() must not raise any exception."""
    streamer = _make_streamer()
    streamer.start()
    assert _wait_ready(streamer), "Buffer did not fill within 5 s"
    streamer.stop()   # must not raise
