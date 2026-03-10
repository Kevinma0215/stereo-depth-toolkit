"""Tests for RollingBuffer."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from stereo_depth.rolling_buffer import RollingBuffer

_FRAME = np.zeros((2, 4, 3), dtype=np.uint8)


def test_empty_buffer():
    buf = RollingBuffer(maxlen=15)
    assert buf.depth == 0
    assert buf.head_index == -1
    assert buf.is_ready is False


def test_push_increments_index():
    buf = RollingBuffer(maxlen=15)
    assert buf.push(_FRAME) == 0
    assert buf.push(_FRAME) == 1
    assert buf.push(_FRAME) == 2


def test_maxlen_enforced():
    buf = RollingBuffer(maxlen=15)
    for _ in range(20):
        buf.push(_FRAME)
    assert buf.depth == 15


def test_is_ready():
    buf = RollingBuffer(maxlen=15)
    for _ in range(14):
        buf.push(_FRAME)
    assert buf.is_ready is False
    buf.push(_FRAME)
    assert buf.is_ready is True


def test_snapshot_returns_correct_index():
    buf = RollingBuffer(maxlen=15)
    for _ in range(5):
        buf.push(_FRAME)
    head, frames = buf.snapshot()
    assert head == 4
    assert len(frames) == 5


def test_snapshot_is_copy():
    buf = RollingBuffer(maxlen=15)
    for _ in range(5):
        buf.push(_FRAME)
    _, frames = buf.snapshot()
    frames.clear()
    assert buf.depth == 5


def test_thread_safety():
    buf = RollingBuffer(maxlen=15)

    def push_50():
        for _ in range(50):
            buf.push(_FRAME)

    t1 = threading.Thread(target=push_50)
    t2 = threading.Thread(target=push_50)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert buf.head_index == 99
    assert buf.depth == 15
