"""Hole-filling post-processor: inpaints NaN/zero depth pixels using INPAINT_NS."""
from __future__ import annotations

import cv2
import numpy as np

from stereo_depth.entities import DepthMap
from stereo_depth.use_cases.ports import IPostProcessor

_MAX_DEPTH_M = 20.0
_SCALE       = 65535.0 / _MAX_DEPTH_M   # float32 metres → uint16


class HoleFillPostProcessor(IPostProcessor):
    """IPostProcessor that fills NaN/zero depth holes via cv2.inpaint (INPAINT_NS).

    cv2.inpaint does not accept float32 directly, so depth is scaled to uint16
    (0–20 m → 0–65535), inpainted, then rescaled back.  Originally-valid pixels
    are preserved exactly.

    Args:
        radius: Inpainting neighbourhood radius in pixels (default 3).
    """

    def __init__(self, radius: int = 3) -> None:
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        self._radius = radius

    def process(self, depth_map: DepthMap) -> DepthMap:
        filled = _inpaint_depth(depth_map.data, self._radius)
        return DepthMap(
            data=filled,
            disparity=depth_map.disparity,
            left_rect=depth_map.left_rect,
            right_rect=depth_map.right_rect,
        )


def _inpaint_depth(depth: np.ndarray, radius: int) -> np.ndarray:
    """Fill NaN holes in a float32 depth map using cv2.inpaint (INPAINT_NS).

    Returns a new float32 array.  Valid pixels are preserved exactly.
    """
    mask = (~np.isfinite(depth)).astype(np.uint8)
    if mask.sum() == 0:
        return depth

    depth_clipped = np.where(np.isfinite(depth), np.clip(depth, 0.0, _MAX_DEPTH_M), 0.0)
    depth_u16     = (depth_clipped * _SCALE).astype(np.uint16)
    depth_u16[mask.astype(bool)] = 0

    filled_u16 = cv2.inpaint(depth_u16, mask, radius, cv2.INPAINT_NS)
    filled     = filled_u16.astype(np.float32) / _SCALE

    # Restore original valid pixels exactly (no float roundtrip error)
    result = depth.copy()
    result[mask.astype(bool)] = filled[mask.astype(bool)]
    return result
