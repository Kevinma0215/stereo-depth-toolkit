"""Convert CalibrationResult to the JSON-compatible dict that retinify expects.

This module does NOT import retinify, so it is safe to use regardless of
whether retinify is installed.

Usage example::

    import json, tempfile, retinify
    from stereo_depth.adapters.calibration.retinify_adapter import (
        calibration_result_to_retinify,
    )

    d = calibration_result_to_retinify(calib)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(d, f)
        calib_params = retinify.load_calibration_parameters(f.name)
"""
from __future__ import annotations

from stereo_depth.entities import CalibrationResult


def calibration_result_to_retinify(calib: CalibrationResult) -> dict:
    """Return a dict that matches the JSON schema accepted by
    ``retinify.load_calibration_parameters()``.

    The produced structure follows the standard stereo calibration convention
    used by retinify's JSON loader:

    .. code-block:: json

        {
          "image_size": {"width": W, "height": H},
          "K1": [[fx,0,cx],[0,fy,cy],[0,0,1]],
          "D1": [k1,k2,p1,p2,k3],
          "K2": ...,
          "D2": ...,
          "R":  3x3 rotation  (right camera w.r.t. left),
          "T":  [tx, ty, tz]  (metres),
          "baseline_m": 0.063
        }

    Rectification matrices (R1/R2/P1/P2/Q) are intentionally omitted because
    retinify computes them internally.

    Args:
        calib: CalibrationResult produced by YamlCalibrationRepo or the
               stereo calibration pipeline.

    Returns:
        A plain dict suitable for ``json.dump``.
    """
    return {
        "image_size": {
            "width":  calib.image_size[0],
            "height": calib.image_size[1],
        },
        "K1": calib.K1.tolist(),
        "D1": calib.D1.tolist(),
        "K2": calib.K2.tolist(),
        "D2": calib.D2.tolist(),
        "R":  calib.R.tolist(),
        "T":  calib.T.tolist(),
        "baseline_m": calib.baseline_m,
    }
