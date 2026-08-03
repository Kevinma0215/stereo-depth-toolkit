"""Undistortion helpers: the K_new pairing invariant.

The one rule that must never break: K/D describe RAW images, while K_new
(with zero distortion) describes UNDISTORTED images. Concretely, for every
destination pixel q of the undistorted image, the remap table must point at
exactly the source pixel that the raw model would have put that ray on.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from stereo_depth.adapters.calibration.mono_calibrator import (
    build_undistort_maps,
    new_camera_matrix,
)

IMG_SIZE = (1280, 720)

K_PINHOLE = np.array([[620.0, 0.0, 645.0],
                      [0.0, 618.0, 355.0],
                      [0.0, 0.0, 1.0]])
D_PINHOLE = np.array([-0.18, 0.04, 0.0008, -0.0006, -0.002])

D_RATIONAL = np.array([-0.18, 0.04, 0.0008, -0.0006, -0.002, 0.01, -0.004, 0.001])

K_FISHEYE = np.array([[470.0, 0.0, 641.0],
                      [0.0, 469.0, 359.0],
                      [0.0, 0.0, 1.0]])
D_FISHEYE = np.array([0.06, -0.012, 0.004, -0.0006])

CASES = [
    ("pinhole", K_PINHOLE, D_PINHOLE),
    ("rational", K_PINHOLE, D_RATIONAL),
    ("fisheye", K_FISHEYE, D_FISHEYE),
]


def _expected_source_pixels(K, D, model, K_new, q):
    """Where raw-image ray through undistorted pixel q actually landed."""
    xn = (q[:, 0] - K_new[0, 2]) / K_new[0, 0]
    yn = (q[:, 1] - K_new[1, 2]) / K_new[1, 1]
    pts3d = np.stack([xn, yn, np.ones_like(xn)], axis=1).astype(np.float64)
    zero = np.zeros(3, dtype=np.float64)
    if model == "fisheye":
        proj, _ = cv2.fisheye.projectPoints(
            pts3d.reshape(1, -1, 3), zero, zero, K, D.reshape(4, 1)
        )
    else:
        proj, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), zero, zero, K, D)
    return proj.reshape(-1, 2)


def _sample_grid(step=40, margin=10):
    w, h = IMG_SIZE
    xs = np.arange(margin, w - margin, step)
    ys = np.arange(margin, h - margin, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


# ---------------------------------------------------------------------------
# The pairing invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,K,D", CASES)
def test_kn_ew_pairs_with_undistorted_images(model, K, D):
    K_new, _roi = new_camera_matrix(K, D, IMG_SIZE, model, alpha=0.0, balance=0.0)
    map_x, map_y = build_undistort_maps(K, D, K_new, IMG_SIZE, model)

    q = _sample_grid()
    expected = _expected_source_pixels(K, D, model, K_new, q)
    actual = np.stack([map_x[q[:, 1], q[:, 0]], map_y[q[:, 1], q[:, 0]]], axis=1)

    err = np.linalg.norm(actual - expected, axis=1)
    assert np.max(err) < 0.5, f"{model}: max pairing error {np.max(err):.3f} px"


@pytest.mark.parametrize("model,K,D", CASES)
def test_pairing_invariant_holds_at_full_alpha(model, K, D):
    """alpha/balance=1 keeps every source pixel; the invariant is unchanged."""
    K_new, _roi = new_camera_matrix(K, D, IMG_SIZE, model, alpha=1.0, balance=1.0)
    map_x, map_y = build_undistort_maps(K, D, K_new, IMG_SIZE, model)

    q = _sample_grid()
    expected = _expected_source_pixels(K, D, model, K_new, q)
    actual = np.stack([map_x[q[:, 1], q[:, 0]], map_y[q[:, 1], q[:, 0]]], axis=1)

    assert np.max(np.linalg.norm(actual - expected, axis=1)) < 0.5


@pytest.mark.parametrize("model,K,D", CASES)
def test_using_raw_K_with_undistorted_images_is_detectably_wrong(model, K, D):
    """Guard against the classic bug: K_new really is a different matrix, so
    pairing raw K with an undistorted image misplaces points substantially."""
    K_new, _roi = new_camera_matrix(K, D, IMG_SIZE, model, alpha=0.0, balance=0.0)

    assert not np.allclose(K, K_new, atol=1.0)

    q = _sample_grid()
    with_new = _expected_source_pixels(K, D, model, K_new, q)
    with_raw = _expected_source_pixels(K, D, model, K, q)
    assert np.max(np.linalg.norm(with_new - with_raw, axis=1)) > 5.0


# ---------------------------------------------------------------------------
# K_new / map properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,K,D", CASES)
def test_maps_have_right_shape_and_are_finite(model, K, D):
    K_new, _ = new_camera_matrix(K, D, IMG_SIZE, model)
    map_x, map_y = build_undistort_maps(K, D, K_new, IMG_SIZE, model)

    w, h = IMG_SIZE
    assert map_x.shape == (h, w)
    assert map_y.shape == (h, w)
    assert np.all(np.isfinite(map_x))
    assert np.all(np.isfinite(map_y))


@pytest.mark.parametrize("model,K,D", CASES)
def test_full_alpha_keeps_more_field_than_cropped(model, K, D):
    """alpha/balance=0 crops to valid pixels (zooms in); =1 keeps everything
    (zooms out), so the focal length must not increase."""
    K_tight, _ = new_camera_matrix(K, D, IMG_SIZE, model, alpha=0.0, balance=0.0)
    K_wide, _ = new_camera_matrix(K, D, IMG_SIZE, model, alpha=1.0, balance=1.0)

    assert K_wide[0, 0] <= K_tight[0, 0] + 1e-6


def test_roi_is_reported_for_pinhole_and_absent_for_fisheye():
    _, roi_pin = new_camera_matrix(K_PINHOLE, D_PINHOLE, IMG_SIZE, "pinhole", alpha=0.0)
    _, roi_fish = new_camera_matrix(K_FISHEYE, D_FISHEYE, IMG_SIZE, "fisheye", balance=0.0)

    assert roi_fish is None
    assert roi_pin is not None
    x, y, w, h = roi_pin
    assert w > 0 and h > 0
    assert 0 <= x < IMG_SIZE[0] and 0 <= y < IMG_SIZE[1]


@pytest.mark.parametrize("model,K,D", CASES)
def test_remap_produces_an_image_of_the_same_size(model, K, D):
    K_new, _ = new_camera_matrix(K, D, IMG_SIZE, model)
    map_x, map_y = build_undistort_maps(K, D, K_new, IMG_SIZE, model)

    w, h = IMG_SIZE
    raw = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    out = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR)

    assert out.shape == raw.shape
    assert out.dtype == np.uint8


def test_straight_line_through_center_stays_straight_after_undistort():
    """A row through the principal point is undistorted onto a row."""
    K, D, model = K_PINHOLE, D_PINHOLE, "pinhole"
    K_new, _ = new_camera_matrix(K, D, IMG_SIZE, model, alpha=0.0)
    map_x, map_y = build_undistort_maps(K, D, K_new, IMG_SIZE, model)

    row = int(round(K_new[1, 2]))
    ys = map_y[row, 100:-100]
    assert np.ptp(ys) < 1.0
