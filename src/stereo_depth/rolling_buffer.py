"""stereo_depth/rolling_buffer.py — thread-safe ring buffer for raw SBS frames."""
from __future__ import annotations

import threading
from collections import deque
from typing import List, Tuple

import numpy as np


class RollingBuffer:
    """Ring buffer storing raw side-by-side stereo frames (H, W*2, 3) uint8 BGR."""

    def __init__(self, maxlen: int = 15) -> None:
        self._deque: deque[np.ndarray] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._head_index: int = -1
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> int:
        """Store one raw SBS frame. Returns the frame index assigned to it."""
        with self._lock:
            self._head_index += 1
            self._deque.append(frame)
            return self._head_index

    def snapshot(self) -> Tuple[int, List[np.ndarray]]:
        """Atomically return (head_index, list_of_frames)."""
        with self._lock:
            return self._head_index, list(self._deque)

    @property
    def depth(self) -> int:
        """Number of frames currently stored (0 to maxlen)."""
        with self._lock:
            return len(self._deque)

    @property
    def head_index(self) -> int:
        """Index of the most recently pushed frame. -1 if empty."""
        with self._lock:
            return self._head_index

    @property
    def is_ready(self) -> bool:
        """True when buffer holds exactly maxlen frames."""
        with self._lock:
            return len(self._deque) == self._maxlen
