"""Static calibration quality evaluation.

Derives all quality metrics purely from a :class:`~stereo_depth.entities.CalibrationResult`
— no images or hardware required.  Synthetic 3-D test points are projected
through the full distortion + rectification pipeline to measure epipolar
alignment residuals.

Typical usage::

    from stereo_depth.adapters.calibration.eval_calibration import evaluate_calibration

    result = evaluate_calibration(calib)
    if not result.passed:
        for w in result.warnings:
            print("WARN:", w)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from stereo_depth.entities import CalibrationResult


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationEvaluation:
    """Quality report for a single :class:`CalibrationResult`.

    Attributes:
        rpe_px:
            Reprojection error stored in the calibration (set by the
            calibrator at calibration time, not recomputed here).
        epipolar_error_mean_px:
            Mean ``|y_left − y_right|`` in rectified pixel space, computed
            across *n_test_points* synthetic 3-D points.  After correct
            stereo rectification all epipolar lines are horizontal, so this
            value should be well below 1 pixel.
        epipolar_error_max_px:
            Maximum epipolar error over the same test points.
        focal_length_asymmetry:
            ``|fx_L − fx_R| / fx_L``.  A well-matched stereo rig should
            have nearly identical focal lengths; values above ~2 % indicate
            a poor calibration or lens mismatch.
        principal_point_offset_px:
            Mean Euclidean distance of each camera's principal point from
            the image centre (in pixels).  Informational; no threshold is
            applied by default.
        baseline_m:
            Copy of ``calib.baseline_m`` for convenient access.
        passed:
            ``True`` when every metric satisfies its threshold as passed to
            :func:`evaluate_calibration`.
        warnings:
            Human-readable descriptions of every violated threshold.
    """

    rpe_px:                    float
    epipolar_error_mean_px:    float
    epipolar_error_max_px:     float
    focal_length_asymmetry:    float
    principal_point_offset_px: float
    baseline_m:                float
    passed:                    bool
    warnings:                  list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_calibration(
    calib: CalibrationResult,
    *,
    max_epipolar_error_px: float = 1.0,
    max_rpe_px:            float = 0.5,
    max_focal_asymmetry:   float = 0.02,
    n_test_points:         int   = 50,
) -> CalibrationEvaluation:
    """Evaluate the quality of a :class:`CalibrationResult` without any images.

    Synthetic 3-D points are scattered in a frustum in front of the left
    camera (depth 0.5–3.0 m, lateral spread ±0.3 m), projected into both
    cameras through their full distortion model, then undistorted and
    rectified with the stored ``R1``/``P1`` and ``R2``/``P2`` matrices.
    The residual vertical distance between corresponding rectified
    projections is the *epipolar error*.

    Args:
        calib:
            The :class:`CalibrationResult` to evaluate.
        max_epipolar_error_px:
            Mean epipolar error threshold in pixels.  Exceeding this adds
            a warning and sets ``passed=False``.
        max_rpe_px:
            Reprojection-error threshold in pixels.
        max_focal_asymmetry:
            Maximum tolerated relative focal-length difference
            ``|fx_L − fx_R| / fx_L``.
        n_test_points:
            Number of random 3-D points used to compute epipolar error.
            More points give a more stable estimate; 50 is sufficient for
            typical use.

    Returns:
        A :class:`CalibrationEvaluation` whose ``passed`` field is ``True``
        only when every metric is within its threshold.
    """
    w, h = calib.image_size
    warns: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. Epipolar alignment                                               #
    # ------------------------------------------------------------------ #
    epi_mean, epi_max = _epipolar_error(calib, n_test_points)

    # ------------------------------------------------------------------ #
    # 2. Focal-length asymmetry                                           #
    # ------------------------------------------------------------------ #
    fx_L = float(calib.K1[0, 0])
    fx_R = float(calib.K2[0, 0])
    focal_asym = abs(fx_L - fx_R) / max(fx_L, 1e-9)

    # ------------------------------------------------------------------ #
    # 3. Principal-point offset from image centre (informational)         #
    # ------------------------------------------------------------------ #
    cx_L, cy_L = float(calib.K1[0, 2]), float(calib.K1[1, 2])
    cx_R, cy_R = float(calib.K2[0, 2]), float(calib.K2[1, 2])
    pp_offset = float(
        np.mean([
            np.hypot(cx_L - w / 2.0, cy_L - h / 2.0),
            np.hypot(cx_R - w / 2.0, cy_R - h / 2.0),
        ])
    )

    # ------------------------------------------------------------------ #
    # 4. Threshold checks → warnings                                      #
    # ------------------------------------------------------------------ #
    if calib.baseline_m <= 0.0:
        warns.append(
            f"baseline_m is not positive ({calib.baseline_m:.6f} m)"
        )
    if calib.rpe_px > max_rpe_px:
        warns.append(
            f"rpe_px {calib.rpe_px:.3f} px exceeds threshold "
            f"{max_rpe_px:.3f} px"
        )
    if epi_mean > max_epipolar_error_px:
        warns.append(
            f"mean epipolar error {epi_mean:.3f} px exceeds threshold "
            f"{max_epipolar_error_px:.3f} px"
        )
    if focal_asym > max_focal_asymmetry:
        warns.append(
            f"focal-length asymmetry {focal_asym * 100:.2f}% exceeds "
            f"threshold {max_focal_asymmetry * 100:.2f}%"
        )

    return CalibrationEvaluation(
        rpe_px=calib.rpe_px,
        epipolar_error_mean_px=epi_mean,
        epipolar_error_max_px=epi_max,
        focal_length_asymmetry=focal_asym,
        principal_point_offset_px=pp_offset,
        baseline_m=calib.baseline_m,
        passed=len(warns) == 0,
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _epipolar_error(
    calib: CalibrationResult,
    n_points: int,
) -> tuple[float, float]:
    """Return ``(mean, max)`` epipolar y-error in pixels.

    For each synthetic 3-D point *X* the pipeline is:

    1. ``cv2.projectPoints(X, rvec=0, tvec=0, K1, D1)`` → distorted left pixel
    2. ``cv2.projectPoints(X, R,       T,      K2, D2)`` → distorted right pixel
    3. ``cv2.undistortPoints(left_px,  K1, D1, R=R1, P=P1)`` → rectified left
    4. ``cv2.undistortPoints(right_px, K2, D2, R=R2, P=P2)`` → rectified right
    5. epipolar error = ``|rect_left_y − rect_right_y|``

    After correct stereo rectification all errors are well below 1 pixel.
    Only points whose unrectified projections land within the image on both
    sides are used; if none qualify, ``(0.0, 0.0)`` is returned.
    """
    rng = np.random.default_rng(42)
    w, h = calib.image_size

    pts3d = np.stack(
        [
            rng.uniform(-0.3, 0.3, n_points),
            rng.uniform(-0.3, 0.3, n_points),
            rng.uniform(0.5,  3.0, n_points),
        ],
        axis=1,
    ).reshape(-1, 1, 3).astype(np.float64)

    zero3   = np.zeros(3, dtype=np.float64)
    rvec_R, _ = cv2.Rodrigues(calib.R.astype(np.float64))

    pts_L, _ = cv2.projectPoints(pts3d, zero3,   zero3,    calib.K1, calib.D1)
    pts_R, _ = cv2.projectPoints(pts3d, rvec_R,  calib.T,  calib.K2, calib.D2)

    pts_L = pts_L.reshape(-1, 2)
    pts_R = pts_R.reshape(-1, 2)

    in_bounds = (
        (pts_L[:, 0] > 0) & (pts_L[:, 0] < w) &
        (pts_L[:, 1] > 0) & (pts_L[:, 1] < h) &
        (pts_R[:, 0] > 0) & (pts_R[:, 0] < w) &
        (pts_R[:, 1] > 0) & (pts_R[:, 1] < h)
    )
    if not np.any(in_bounds):
        return 0.0, 0.0

    pts_L = pts_L[in_bounds].reshape(-1, 1, 2).astype(np.float32)
    pts_R = pts_R[in_bounds].reshape(-1, 1, 2).astype(np.float32)

    rect_L = cv2.undistortPoints(
        pts_L, calib.K1, calib.D1, R=calib.R1, P=calib.P1
    ).reshape(-1, 2)
    rect_R = cv2.undistortPoints(
        pts_R, calib.K2, calib.D2, R=calib.R2, P=calib.P2
    ).reshape(-1, 2)

    y_errors = np.abs(rect_L[:, 1] - rect_R[:, 1])
    return float(np.mean(y_errors)), float(np.max(y_errors))
