"""Isaac Sim / USD parameter conversion — algebraic round-trips.

Every assertion here inverts the documented formula, so a sign slip or a
swapped width/height cannot pass.
"""
from __future__ import annotations

import numpy as np
import pytest

from stereo_depth.adapters.calibration.isaac_export import (
    DEFAULT_HORIZONTAL_APERTURE_MM,
    distortion_block,
    opencv_fisheye_params,
    opencv_pinhole_params,
    usd_camera_params,
)

IMG_SIZE = (1280, 720)
K = np.array([[620.0, 0.0, 645.0],
              [0.0, 618.0, 355.0],
              [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# USD classic camera attributes
# ---------------------------------------------------------------------------

def test_focal_length_inverts_to_fx():
    p = usd_camera_params(K, IMG_SIZE)
    w = IMG_SIZE[0]

    fx_back = p["focal_length_mm"] * w / p["horizontal_aperture_mm"]
    assert fx_back == pytest.approx(K[0, 0])


def test_vertical_aperture_inverts_to_fy():
    """fy = focalLength * H / verticalAperture must hold exactly, which is
    what makes a non-square pixel aspect survive the conversion."""
    p = usd_camera_params(K, IMG_SIZE)
    h = IMG_SIZE[1]

    fy_back = p["focal_length_mm"] * h / p["vertical_aperture_mm"]
    assert fy_back == pytest.approx(K[1, 1])


def test_aperture_offsets_invert_to_principal_point():
    p = usd_camera_params(K, IMG_SIZE)
    w, h = IMG_SIZE

    cx_back = p["horizontal_aperture_offset_mm"] * w / p["horizontal_aperture_mm"] + w / 2
    cy_back = h / 2 - p["vertical_aperture_offset_mm"] * h / p["vertical_aperture_mm"]
    assert cx_back == pytest.approx(K[0, 2])
    assert cy_back == pytest.approx(K[1, 2])


def test_principal_point_right_of_centre_gives_positive_horizontal_offset():
    k = K.copy()
    k[0, 2] = IMG_SIZE[0] / 2 + 30.0
    assert usd_camera_params(k, IMG_SIZE)["horizontal_aperture_offset_mm"] > 0

    k[0, 2] = IMG_SIZE[0] / 2 - 30.0
    assert usd_camera_params(k, IMG_SIZE)["horizontal_aperture_offset_mm"] < 0


def test_principal_point_below_centre_gives_negative_vertical_offset():
    """Image y grows downward while USD film y grows upward — the sign flips."""
    k = K.copy()
    k[1, 2] = IMG_SIZE[1] / 2 + 30.0        # lower in the image
    assert usd_camera_params(k, IMG_SIZE)["vertical_aperture_offset_mm"] < 0

    k[1, 2] = IMG_SIZE[1] / 2 - 30.0        # higher in the image
    assert usd_camera_params(k, IMG_SIZE)["vertical_aperture_offset_mm"] > 0


def test_centered_principal_point_gives_zero_offsets():
    k = np.array([[620.0, 0.0, IMG_SIZE[0] / 2],
                  [0.0, 620.0, IMG_SIZE[1] / 2],
                  [0.0, 0.0, 1.0]])
    p = usd_camera_params(k, IMG_SIZE)

    assert p["horizontal_aperture_offset_mm"] == pytest.approx(0.0)
    assert p["vertical_aperture_offset_mm"] == pytest.approx(0.0)


def test_square_pixels_give_aperture_ratio_equal_to_image_aspect():
    k = np.array([[620.0, 0.0, 640.0],
                  [0.0, 620.0, 360.0],
                  [0.0, 0.0, 1.0]])
    p = usd_camera_params(k, IMG_SIZE)

    assert (p["horizontal_aperture_mm"] / p["vertical_aperture_mm"]) == pytest.approx(
        IMG_SIZE[0] / IMG_SIZE[1]
    )


def test_custom_sensor_width_rescales_focal_length_proportionally():
    """Only the focal/aperture ratio matters, so doubling the assumed sensor
    width must double the reported focal length."""
    a = usd_camera_params(K, IMG_SIZE, horizontal_aperture_mm=10.0)
    b = usd_camera_params(K, IMG_SIZE, horizontal_aperture_mm=20.0)

    assert b["focal_length_mm"] == pytest.approx(2 * a["focal_length_mm"])
    assert b["vertical_aperture_mm"] == pytest.approx(2 * a["vertical_aperture_mm"])


def test_default_aperture_is_isaac_sim_default():
    assert usd_camera_params(K, IMG_SIZE)["horizontal_aperture_mm"] == pytest.approx(
        DEFAULT_HORIZONTAL_APERTURE_MM
    )


def test_all_usd_values_are_finite_floats():
    p = usd_camera_params(K, IMG_SIZE)
    assert set(p) == {
        "focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm",
        "horizontal_aperture_offset_mm", "vertical_aperture_offset_mm",
    }
    for key, val in p.items():
        assert isinstance(val, float), key
        assert np.isfinite(val), key


# ---------------------------------------------------------------------------
# OpenCV distortion blocks
# ---------------------------------------------------------------------------

def test_pinhole_block_zero_fills_higher_order_terms():
    d5 = np.array([-0.18, 0.04, 0.0008, -0.0006, -0.002])
    p = opencv_pinhole_params(K, d5, IMG_SIZE)

    assert p["k1"] == pytest.approx(-0.18)
    assert p["k2"] == pytest.approx(0.04)
    assert p["p1"] == pytest.approx(0.0008)
    assert p["p2"] == pytest.approx(-0.0006)
    assert p["k3"] == pytest.approx(-0.002)
    assert p["k4"] == p["k5"] == p["k6"] == 0.0


def test_pinhole_block_preserves_rational_coefficients():
    d8 = np.array([-0.18, 0.04, 0.0008, -0.0006, -0.002, 0.01, -0.004, 0.001])
    p = opencv_pinhole_params(K, d8, IMG_SIZE)

    assert p["k4"] == pytest.approx(0.01)
    assert p["k5"] == pytest.approx(-0.004)
    assert p["k6"] == pytest.approx(0.001)


def test_blocks_carry_pixel_intrinsics_and_nominal_size():
    for params in (
        opencv_pinhole_params(K, np.zeros(5), IMG_SIZE),
        opencv_fisheye_params(K, np.zeros(4), IMG_SIZE),
    ):
        assert params["fx"] == pytest.approx(K[0, 0])
        assert params["fy"] == pytest.approx(K[1, 1])
        assert params["cx"] == pytest.approx(K[0, 2])
        assert params["cy"] == pytest.approx(K[1, 2])
        assert params["nominal_width"] == IMG_SIZE[0]
        assert params["nominal_height"] == IMG_SIZE[1]


def test_fisheye_block_has_exactly_four_coefficients():
    d4 = np.array([0.06, -0.012, 0.004, -0.0006])
    p = opencv_fisheye_params(K, d4, IMG_SIZE)

    assert (p["k1"], p["k2"], p["k3"], p["k4"]) == pytest.approx(tuple(d4))
    assert "p1" not in p and "k5" not in p


@pytest.mark.parametrize("model,expected", [
    ("pinhole", "opencv_pinhole"),
    ("rational", "opencv_pinhole"),
    ("fisheye", "opencv_fisheye"),
])
def test_distortion_block_picks_the_right_usd_schema(model, expected):
    d = np.zeros(4 if model == "fisheye" else 5)
    name, params = distortion_block(model, K, d, IMG_SIZE)

    assert name == expected
    assert params["fx"] == pytest.approx(K[0, 0])
