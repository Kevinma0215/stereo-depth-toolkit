"""Convert OpenCV intrinsics into Isaac Sim / Omniverse USD camera parameters.

Two independent ways to put a real camera into Isaac Sim:

  A. Let the sim model the distortion. Fill a USD lens-distortion block
     (``OpenCvPinhole`` with k1..k6,p1,p2 or ``OpenCvFisheye`` with k1..k4)
     using the RAW K/D. The sim then renders geometrically like the real
     camera, and the real pipeline consumes RAW frames unchanged.

  B. Undistort on both sides. Give the sim a plain pinhole camera built from
     ``K_new`` with zero distortion, and run every real frame through the
     undistort remap first.

Never mix the two: raw frames go with K/D, undistorted frames go with K_new.

Conventions used below
----------------------
* All aperture and focal quantities are in MILLIMETRES; fx, fy, cx, cy are in
  PIXELS.
* ``horizontal_aperture_mm`` is the sensor width. Only the ratio
  focalLength/aperture affects projection, so the Isaac Sim default of
  20.955 mm is a safe stand-in when the true sensor size is unknown.
* USD film coordinates have +Y up while image coordinates have +y down, so
  the vertical aperture offset carries a sign flip.
"""
from __future__ import annotations

import numpy as np

# Isaac Sim's default camera horizontal aperture (mm).
DEFAULT_HORIZONTAL_APERTURE_MM = 20.955


def usd_camera_params(
    K: np.ndarray,
    image_size: tuple[int, int],
    *,
    horizontal_aperture_mm: float = DEFAULT_HORIZONTAL_APERTURE_MM,
) -> dict:
    """Classic USD camera attributes for a distortion-free pinhole camera.

        focalLength              = fx * Ah / W
        verticalAperture         = Ah * (H * fx) / (W * fy)
        horizontalApertureOffset = (cx - W/2) * Ah / W
        verticalApertureOffset   = (H/2 - cy) * Av / H

    ``verticalAperture`` is derived (rather than assumed square) so that
    ``fy = focalLength * H / Av`` holds exactly even when fx != fy.
    """
    w, h = int(image_size[0]), int(image_size[1])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    ah = float(horizontal_aperture_mm)

    focal = fx * ah / w
    av = ah * (h * fx) / (w * fy)

    return {
        "focal_length_mm": focal,
        "horizontal_aperture_mm": ah,
        "vertical_aperture_mm": av,
        "horizontal_aperture_offset_mm": (cx - w / 2.0) * ah / w,
        "vertical_aperture_offset_mm": (h / 2.0 - cy) * av / h,
    }


def _fxfycxcy(K: np.ndarray) -> dict:
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }


def opencv_pinhole_params(
    K: np.ndarray, D: np.ndarray, image_size: tuple[int, int]
) -> dict:
    """USD ``OpenCvPinhole`` lens-distortion block.

    Accepts a 5-coefficient (k1,k2,p1,p2,k3) or 8-coefficient
    (k1,k2,p1,p2,k3,k4,k5,k6) OpenCV D vector; absent higher-order terms are
    emitted as zero so the block always has the full parameter set.
    """
    d = np.asarray(D, dtype=np.float64).ravel()
    if d.size < 8:
        d = np.pad(d, (0, 8 - d.size))

    out = _fxfycxcy(K)
    out.update({
        "nominal_width": int(image_size[0]),
        "nominal_height": int(image_size[1]),
        # OpenCV D order is (k1, k2, p1, p2, k3, k4, k5, k6)
        "k1": float(d[0]), "k2": float(d[1]),
        "p1": float(d[2]), "p2": float(d[3]),
        "k3": float(d[4]), "k4": float(d[5]),
        "k5": float(d[6]), "k6": float(d[7]),
    })
    return out


def opencv_fisheye_params(
    K: np.ndarray, D: np.ndarray, image_size: tuple[int, int]
) -> dict:
    """USD ``OpenCvFisheye`` lens-distortion block (equidistant, k1..k4)."""
    d = np.asarray(D, dtype=np.float64).ravel()
    if d.size < 4:
        d = np.pad(d, (0, 4 - d.size))

    out = _fxfycxcy(K)
    out.update({
        "nominal_width": int(image_size[0]),
        "nominal_height": int(image_size[1]),
        "k1": float(d[0]), "k2": float(d[1]),
        "k3": float(d[2]), "k4": float(d[3]),
    })
    return out


def distortion_block(
    model: str, K: np.ndarray, D: np.ndarray, image_size: tuple[int, int]
) -> tuple[str, dict]:
    """(usd_block_name, params) for the given OpenCV distortion model."""
    if model == "fisheye":
        return "opencv_fisheye", opencv_fisheye_params(K, D, image_size)
    return "opencv_pinhole", opencv_pinhole_params(K, D, image_size)
