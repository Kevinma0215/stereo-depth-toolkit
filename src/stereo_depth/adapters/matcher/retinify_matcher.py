"""Retinify-backed IDisparityMatcher.

Raises ``ImportError`` at module import time if the ``retinify`` package is not
installed.  Install instructions are included in the error message.
"""
from __future__ import annotations
import numpy as np

try:
    import retinify as _retinify
except ImportError as _exc:
    raise ImportError(
        "The 'retinify' package is not installed.\n"
        "  Install: pip install retinify\n"
        "  Requirements: CUDA Toolkit, cuDNN, TensorRT\n"
        "  See: https://github.com/retinify/retinify"
    ) from _exc

from stereo_depth.use_cases.ports import IDisparityMatcher

_MODE_MAP: dict[str, "_retinify.DepthMode"] = {
    "fast":     _retinify.DepthMode.FAST,
    "balanced": _retinify.DepthMode.BALANCED,
    "accurate": _retinify.DepthMode.ACCURATE,
}

_VALID_MODES = tuple(_MODE_MAP)


class RetinifyMatcher(IDisparityMatcher):
    """IDisparityMatcher backed by the Retinify TensorRT AI stereo engine.

    Pass **pre-rectified** images to ``compute()``; Retinify's internal
    rectification is skipped when no ``calibration_parameters`` are provided
    to ``initialize()``.

    Args:
        width:  Image width in pixels (must match the images passed to compute()).
        height: Image height in pixels.
        mode:   Depth quality mode — 'fast' | 'balanced' (default) | 'accurate'.
                Maps to retinify.DepthMode.{FAST,BALANCED,ACCURATE}.
    """

    def __init__(
        self,
        width: int,
        height: int,
        mode: str = "balanced",
    ) -> None:
        mode_lc = mode.lower()
        if mode_lc not in _MODE_MAP:
            raise ValueError(
                f"Unknown Retinify mode '{mode}'. "
                f"Valid options: {_VALID_MODES}"
            )
        self._pipe = _retinify.Pipeline()
        self._pipe.initialize(width, height, depth_mode=_MODE_MAP[mode_lc])

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Run Retinify inference on a rectified stereo pair.

        Args:
            left:  uint8 image, shape (H, W) or (H, W, 3).
            right: uint8 image, same shape as left.

        Returns:
            float32 disparity map, shape (H, W).
        """
        self._pipe.execute(left, right)
        disp = self._pipe.retrieve_disparity()
        return disp.astype(np.float32)
