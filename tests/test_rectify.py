"""Tests for OpenCVRectifier.

All tests are hardware-independent: calibration is synthesised with
cv2.stereoRectify so no real camera or saved data is needed.
"""
from __future__ import annotations
import numpy as np
import cv2
import pytest

from stereo_depth.entities import FramePair, RectifiedPair, CalibrationResult
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_calib(
    w: int = 640,
    h: int = 480,
    f: float = 500.0,
    baseline: float = 0.06,
    rot_deg: float = 2.0,
) -> CalibrationResult:
    """Synthetic stereo rig with slight y-axis rotation between cameras."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)

    angle = np.deg2rad(rot_deg)
    R = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    T = np.array([-baseline, 0.0, 0.0], dtype=np.float64)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, D, K, D, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    return CalibrationResult(
        image_size=(w, h),
        K1=K.copy(), D1=D.copy(),
        K2=K.copy(), D2=D.copy(),
        R=R, T=T,
        baseline_m=float(np.linalg.norm(T)),
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        rpe_px=0.0,
    )


def _draw_marker(img: np.ndarray, cx: float, cy: float, radius: int = 12) -> None:
    cv2.circle(img, (int(round(cx)), int(round(cy))), radius, (255, 255, 255), -1)


def _find_centroid(img: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    if M["m00"] == 0:
        return None, None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_type_and_shape():
    """rectify() must return a RectifiedPair with the same spatial dimensions."""
    calib = _make_synthetic_calib()
    w, h = calib.image_size

    pair = FramePair(
        left=np.zeros((h, w, 3), dtype=np.uint8),
        right=np.zeros((h, w, 3), dtype=np.uint8),
    )
    rect = OpenCVRectifier().rectify(pair, calib)

    assert isinstance(rect, RectifiedPair)
    assert rect.left.shape == (h, w, 3)
    assert rect.right.shape == (h, w, 3)
    assert rect.left.dtype == np.uint8
    assert rect.right.dtype == np.uint8


def test_epipolar_alignment():
    """A 3D point projected to both cameras must land on the same row
    (within 2 px) after rectification."""
    calib = _make_synthetic_calib(rot_deg=2.0)
    w, h = calib.image_size

    # 3D point in left-camera frame
    X_cam = np.array([0.05, 0.04, 1.2], dtype=np.float64)

    # Unrectified left projection:  K1 @ X_cam
    pt_L = calib.K1 @ X_cam
    u_L, v_L = pt_L[0] / pt_L[2], pt_L[1] / pt_L[2]

    # Unrectified right projection: K2 @ (R @ X_cam + T)
    X_right = calib.R @ X_cam + calib.T
    pt_R = calib.K2 @ X_right
    u_R, v_R = pt_R[0] / pt_R[2], pt_R[1] / pt_R[2]

    margin = 20
    assert margin < u_L < w - margin and margin < v_L < h - margin, (
        f"left projection ({u_L:.1f}, {v_L:.1f}) out of image bounds"
    )
    assert margin < u_R < w - margin and margin < v_R < h - margin, (
        f"right projection ({u_R:.1f}, {v_R:.1f}) out of image bounds"
    )

    # Draw markers on black images
    left_img = np.zeros((h, w, 3), dtype=np.uint8)
    right_img = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_marker(left_img, u_L, v_L)
    _draw_marker(right_img, u_R, v_R)

    rect = OpenCVRectifier().rectify(
        FramePair(left=left_img, right=right_img), calib
    )

    cx_L, cy_L = _find_centroid(rect.left)
    cx_R, cy_R = _find_centroid(rect.right)

    assert cx_L is not None, "marker not found in rectified left image"
    assert cx_R is not None, "marker not found in rectified right image"

    epipolar_error = abs(cy_L - cy_R)
    assert epipolar_error < 2.0, (
        f"epipolar error {epipolar_error:.2f} px exceeds 2 px threshold"
    )


def test_maps_cached_across_calls():
    """Passing the same CalibrationResult object twice must not rebuild maps."""
    calib = _make_synthetic_calib()
    w, h = calib.image_size

    rectifier = OpenCVRectifier()
    pair = FramePair(
        left=np.zeros((h, w, 3), dtype=np.uint8),
        right=np.zeros((h, w, 3), dtype=np.uint8),
    )

    rectifier.rectify(pair, calib)
    cache_id_after_first = rectifier._cache_id

    rectifier.rectify(pair, calib)
    assert rectifier._cache_id == cache_id_after_first  # no rebuild
