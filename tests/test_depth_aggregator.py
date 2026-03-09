"""Tests for DepthAggregator.

All tests are hardware-independent (synthetic data only).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from stereo_depth.entities.calibration import CalibrationResult
from stereo_depth.infrastructure.depth_aggregator import (
    DepthAggregator,
    DepthAggregatorError,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def synthetic_calib() -> CalibrationResult:
    """CalibrationResult with realistic HBVCAM-W202011HD values.

    fx=fy=480, cx=320, cy=240, baseline=0.06 m, image_size=(640, 480).
    Q matrix built analytically from these values.
    """
    fx = fy = 480.0
    cx, cy = 320.0, 240.0
    baseline = 0.06

    K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K2 = K1.copy()
    D1 = np.zeros(5, dtype=np.float64)
    D2 = np.zeros(5, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    T = np.array([-baseline, 0.0, 0.0], dtype=np.float64)
    R1 = np.eye(3, dtype=np.float64)
    R2 = np.eye(3, dtype=np.float64)
    P1 = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
    P2 = np.array(
        [[fx, 0, cx, -fx * baseline], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64
    )
    # Standard stereoRectify Q when cx_left == cx_right:
    # p = Q @ [u, v, d, 1]  =>  XYZ = p[:3] / p[3]
    # Z = fx * baseline / d
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
        image_size=(640, 480),
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T,
        baseline_m=baseline,
        R1=R1, R2=R2, P1=P1, P2=P2,
        Q=Q,
        rpe_px=0.0,
    )


def flat_disparity(depth_m: float, shape: tuple[int, int] = (480, 640)) -> np.ndarray:
    """Float32 disparity map where every pixel corresponds to depth_m metres."""
    fx = 480.0
    baseline = 0.06
    d = fx * baseline / depth_m
    return np.full(shape, d, dtype=np.float32)


def noisy_disparity(
    depth_m: float,
    hole_fraction: float,
    shape: tuple[int, int] = (480, 640),
    seed: int = 0,
) -> np.ndarray:
    """flat_disparity with hole_fraction of pixels randomly set to 0 (invalid)."""
    disp = flat_disparity(depth_m, shape)
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) < hole_fraction
    disp[mask] = 0.0
    return disp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_push_fills_buffer():
    calib = synthetic_calib()
    agg = DepthAggregator(calib, buffer_size=10)
    for _ in range(10):
        agg.push(flat_disparity(1.0))
    assert agg.stats().buffer_fill == 10


def test_query_before_push_raises():
    calib = synthetic_calib()
    agg = DepthAggregator(calib)
    with pytest.raises(DepthAggregatorError):
        agg.query([(320, 240)])


def test_single_frame_clean_disparity():
    calib = synthetic_calib()
    agg = DepthAggregator(calib, min_confidence=0.0)
    agg.push(flat_disparity(1.0))

    results = agg.query([(320, 240)])
    wp = results[0]

    assert wp.xyz_m is not None
    assert abs(wp.xyz_m[2] - 1.0) / 1.0 < 0.02, (
        f"Z depth {wp.xyz_m[2]:.4f} m not within 2% of 1.0 m"
    )
    assert wp.confidence == pytest.approx(1.0)


def test_temporal_fusion_improves_confidence():
    calib = synthetic_calib()
    frames = [noisy_disparity(1.0, hole_fraction=0.5, seed=i) for i in range(10)]

    agg = DepthAggregator(calib, buffer_size=10, min_confidence=0.0)
    for f in frames:
        agg.push(f)

    # Find pixels that are holes in frame 0 but valid in >= 8 of frames 1–9
    frame0 = frames[0]
    H, W = frame0.shape
    valid_in_others = sum((f > 0).astype(np.int32) for f in frames[1:])
    candidates = (frame0 == 0) & (valid_in_others >= 8)
    ys, xs = np.where(candidates)
    assert len(ys) >= 5, (
        f"Not enough test pixels: found {len(ys)}, need >= 5"
    )
    test_pixels = [(int(xs[i]), int(ys[i])) for i in range(5)]

    results = agg.query(test_pixels)
    mean_conf = float(np.mean([r.confidence for r in results]))
    assert mean_conf > 0.5, f"mean confidence {mean_conf:.3f} not > 0.5"


def test_modal_depth_rejects_outliers():
    calib = synthetic_calib()
    agg = DepthAggregator(calib, buffer_size=10, min_confidence=0.0)

    for _ in range(8):
        agg.push(flat_disparity(1.0))
    for _ in range(2):
        agg.push(flat_disparity(3.0))

    results = agg.query([(320, 240)])
    wp = results[0]

    assert wp.xyz_m is not None
    assert abs(wp.xyz_m[2] - 1.0) / 1.0 < 0.05, (
        f"Z depth {wp.xyz_m[2]:.4f} m not within 5% of 1.0 m — "
        "modal should have selected the majority depth"
    )


def test_T_cam_to_base_transform():
    calib = synthetic_calib()
    # Pure translation: +0.5 m in X
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = 0.5

    agg = DepthAggregator(calib, min_confidence=0.0, T_cam_to_base=T)
    agg.push(flat_disparity(1.0))

    results = agg.query([(320, 240)])
    wp = results[0]

    assert wp.frame == "robot_base"
    assert wp.xyz_m is not None

    # Camera XYZ at center pixel: X=0, Y=0, Z≈1.0 m
    # After +0.5 m X translation: X≈0.5 m
    assert abs(wp.xyz_m[0] - 0.5) < 0.001, (
        f"robot_base X = {wp.xyz_m[0]:.4f} m, expected ≈0.5 m"
    )


def test_low_confidence_returns_none():
    calib = synthetic_calib()
    # Use fixed seeds so the test is deterministic
    agg = DepthAggregator(calib, buffer_size=10, min_confidence=0.7)
    for i in range(10):
        agg.push(noisy_disparity(1.0, hole_fraction=0.95, seed=i + 100))

    results = agg.query([(320, 240)])
    wp = results[0]

    assert wp.xyz_m is None, "Expected xyz_m=None for very low confidence"
    assert wp.confidence < 0.3, (
        f"confidence {wp.confidence:.3f} should be < 0.3 with 95% hole fraction"
    )


def test_thread_safety():
    calib = synthetic_calib()
    agg = DepthAggregator(calib, buffer_size=10, min_confidence=0.0)

    errors: list[Exception] = []
    bad_depths: list[float] = []

    def push_worker():
        try:
            for _ in range(100):
                agg.push(flat_disparity(1.0))
        except Exception as e:
            errors.append(e)

    def query_worker():
        try:
            for _ in range(50):
                try:
                    results = agg.query([(320, 240)])
                    for r in results:
                        if r.xyz_m is not None:
                            z = r.xyz_m[2]
                            if abs(z - 1.0) / 1.0 >= 0.05:
                                bad_depths.append(z)
                except DepthAggregatorError:
                    pass  # called before first push — acceptable race
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=push_worker),
        threading.Thread(target=push_worker),
        threading.Thread(target=query_worker),
        threading.Thread(target=query_worker),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert not bad_depths, (
        f"Got {len(bad_depths)} query results with Z outside 5% tolerance"
    )


def test_stats_latency_reported():
    calib = synthetic_calib()
    agg = DepthAggregator(calib, min_confidence=0.0)
    for _ in range(5):
        agg.push(flat_disparity(1.0))
    agg.query([(320, 240)])

    s = agg.stats()
    assert s.push_latency_ms > 0, "push_latency_ms should be > 0"
    assert s.query_latency_ms > 0, "query_latency_ms should be > 0"


def test_performance_benchmark():
    """100 queries × 6 waypoints must complete in < 10 ms mean."""
    calib = synthetic_calib()
    buffer_size = 10
    agg = DepthAggregator(calib, buffer_size=buffer_size, min_confidence=0.0)

    for _ in range(buffer_size):
        agg.push(flat_disparity(1.0))

    waypoints = [
        (320, 240), (160, 120), (480, 120),
        (160, 360), (480, 360), (320, 360),
    ]

    n_queries = 100
    latencies: list[float] = []
    for _ in range(n_queries):
        t0 = time.perf_counter()
        agg.query(waypoints)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies_arr = np.array(latencies)
    mean_ms = float(np.mean(latencies_arr))
    p95_ms = float(np.percentile(latencies_arr, 95))
    min_ms = float(np.min(latencies_arr))
    max_ms = float(np.max(latencies_arr))
    verdict = "PASS" if mean_ms < 10.0 else "FAIL"

    print(
        f"\n  DepthAggregator benchmark"
        f"\n  {'─' * 25}"
        f"\n  frames in buffer : {buffer_size}"
        f"\n  queries run      : {n_queries}"
        f"\n  waypoints/query  : {len(waypoints)}"
        f"\n  mean latency     : {mean_ms:.2f} ms"
        f"\n  p95 latency      : {p95_ms:.2f} ms"
        f"\n  min/max          : {min_ms:.2f} / {max_ms:.2f} ms"
        f"\n  verdict          : {verdict} (threshold: 10ms)"
    )

    assert mean_ms < 10.0, (
        f"Mean query latency {mean_ms:.2f} ms exceeds 10 ms threshold"
    )
