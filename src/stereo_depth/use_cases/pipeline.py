from __future__ import annotations

from stereo_depth.entities import FramePair, DepthMap, CalibrationResult
from stereo_depth.use_cases.ports import IRectifier, IDisparityMatcher, IDepthEstimator


class StereoPipeline:
    """Wires together a rectifier, a disparity matcher, and a depth estimator.

    The pipeline holds a single CalibrationResult for its lifetime.  Pass the
    same instance across all ``process()`` calls so that the rectifier can
    cache its undistort/rectify maps efficiently.

    Data flow::

        FramePair
          → IRectifier.rectify()   → RectifiedPair
          → IDisparityMatcher.compute() → disparity (H×W float32)
          → IDepthEstimator.to_depth()  → DepthMap (left_rect=None)
          → attach RectifiedPair.left as DepthMap.left_rect
          → return DepthMap
    """

    def __init__(
        self,
        rectifier: IRectifier,
        matcher: IDisparityMatcher,
        depth_estimator: IDepthEstimator,
        calib: CalibrationResult,
    ) -> None:
        self._rectifier = rectifier
        self._matcher = matcher
        self._depth_estimator = depth_estimator
        self._calib = calib

    def process(self, pair: FramePair) -> DepthMap:
        rect = self._rectifier.rectify(pair, self._calib)
        disp = self._matcher.compute(rect.left, rect.right)
        depth_map = self._depth_estimator.to_depth(disp, self._calib)
        return DepthMap(
            data=depth_map.data,
            disparity=depth_map.disparity,
            left_rect=rect.left,
            right_rect=rect.right,
        )
