"""Background-thread-safe disparity accumulator for VLA waypoint planning."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from stereo_depth.entities.calibration import CalibrationResult


class DepthAggregatorError(Exception):
    """Raised when DepthAggregator is used incorrectly."""


@dataclass
class WaypointDepth:
    pixel: tuple[int, int]
    xyz_m: tuple[float, float, float] | None  # None if confidence below threshold
    confidence: float                          # fraction of valid frames (0.0–1.0)
    frame: str                                 # "camera" or "robot_base"


@dataclass
class AggregatorStats:
    buffer_fill: int        # how many frames currently in buffer (0–buffer_size)
    mean_confidence: float  # average confidence across last query
    query_latency_ms: float # time taken for last query() call
    push_latency_ms: float  # time taken for last push() call


class DepthAggregator:
    """
    Accumulates disparity frames in a background thread and answers
    sparse 3D point queries for VLA waypoint planning.

    Phase 1: outputs XYZ in camera frame.
    Phase 2: if T_cam_to_base (4x4 np.ndarray) is provided at init,
             automatically transforms output to robot base frame.
    """

    def __init__(
        self,
        calib: CalibrationResult,
        buffer_size: int = 10,
        min_confidence: float = 0.7,
        roi_radius: int = 8,
        T_cam_to_base: np.ndarray | None = None,
    ) -> None:
        self._Q = calib.Q.astype(np.float64)
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self._min_confidence = min_confidence
        self._roi_radius = roi_radius
        self._T = T_cam_to_base
        self._lock = threading.RLock()
        self._push_latency_ms: float = 0.0
        self._query_latency_ms: float = 0.0
        self._mean_confidence: float = 0.0

    def push(self, disparity: np.ndarray) -> None:
        """Add a new float32 disparity frame to the ring buffer. Thread-safe."""
        t0 = time.perf_counter()
        disp = disparity.astype(np.float32)
        with self._lock:
            self._buffer.append(disp)
        self._push_latency_ms = (time.perf_counter() - t0) * 1000

    def query(self, pixels: list[tuple[int, int]]) -> list[WaypointDepth]:
        """
        For each (u, v) pixel, return XYZ + confidence.
        Uses modal depth within roi_radius patch across all buffered frames.
        Thread-safe — can be called from a different thread than push().
        """
        t0 = time.perf_counter()
        with self._lock:
            if not self._buffer:
                raise DepthAggregatorError("query() called before any push()")
            frames = list(self._buffer)

        results: list[WaypointDepth] = []
        for u, v in pixels:
            results.append(self._query_pixel(u, v, frames))

        self._query_latency_ms = (time.perf_counter() - t0) * 1000
        self._mean_confidence = (
            float(np.mean([r.confidence for r in results])) if results else 0.0
        )
        return results

    def stats(self) -> AggregatorStats:
        """Return performance diagnostics."""
        with self._lock:
            fill = len(self._buffer)
        return AggregatorStats(
            buffer_fill=fill,
            mean_confidence=self._mean_confidence,
            query_latency_ms=self._query_latency_ms,
            push_latency_ms=self._push_latency_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_pixel(
        self, u: int, v: int, frames: list[np.ndarray]
    ) -> WaypointDepth:
        frame_name = "camera" if self._T is None else "robot_base"
        n_frames = len(frames)
        r = self._roi_radius

        all_vals: list[np.ndarray] = []
        valid_frame_count = 0

        for disp_frame in frames:
            h, w = disp_frame.shape
            # Confidence: fraction of frames where the center pixel is valid
            if 0 <= v < h and 0 <= u < w and disp_frame[v, u] > 0:
                valid_frame_count += 1
            # Gather all valid disparities from the patch for modal estimation
            r0 = max(0, v - r)
            r1 = min(h, v + r + 1)
            c0 = max(0, u - r)
            c1 = min(w, u + r + 1)
            patch = disp_frame[r0:r1, c0:c1]
            valid = patch[patch > 0]
            if valid.size:
                all_vals.append(valid.ravel())

        confidence = valid_frame_count / n_frames

        if not all_vals or confidence < self._min_confidence:
            return WaypointDepth(
                pixel=(u, v), xyz_m=None, confidence=confidence, frame=frame_name
            )

        vals = np.concatenate(all_vals)
        d_modal = _modal_disparity(vals)
        xyz = _disparity_to_xyz(self._Q, u, v, d_modal)

        if xyz is None:
            return WaypointDepth(
                pixel=(u, v), xyz_m=None, confidence=confidence, frame=frame_name
            )

        if self._T is not None:
            p = self._T @ np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
            xyz = (float(p[0]), float(p[1]), float(p[2]))

        return WaypointDepth(
            pixel=(u, v), xyz_m=xyz, confidence=confidence, frame=frame_name
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _modal_disparity(vals: np.ndarray, bin_width: float = 0.5) -> float:
    """Return the centre of the most-populated 0.5-unit-wide disparity bucket."""
    vmin = float(vals.min())
    bins = ((vals - vmin) / bin_width).astype(np.int32)
    counts = np.bincount(bins)
    modal_bin = int(np.argmax(counts))
    return vmin + (modal_bin + 0.5) * bin_width


def _disparity_to_xyz(
    Q: np.ndarray, u: int, v: int, d: float
) -> tuple[float, float, float] | None:
    """Convert (u, v, d) to metric XYZ using the Q reprojection matrix."""
    if d <= 0:
        return None
    p = Q @ np.array([u, v, d, 1.0], dtype=np.float64)
    w = p[3]
    if abs(w) < 1e-9:
        return None
    xyz = p[:3] / w
    if not np.all(np.isfinite(xyz)):
        return None
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
