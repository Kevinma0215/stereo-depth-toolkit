from __future__ import annotations

from typing import Iterator, Optional

from stereo_depth.entities import FramePair, DepthMap, CalibrationResult
from stereo_depth.use_cases.ports import (
    ICameraSource,
    IRectifier,
    IDisparityMatcher,
    IDepthEstimator,
    IPostProcessor,
)


class StereoPipeline:
    """Wires together a rectifier, a disparity matcher, a depth estimator,
    and an optional chain of post-processors.

    The pipeline holds a single CalibrationResult for its lifetime.  Pass the
    same instance across all ``process()`` calls so that the rectifier can
    cache its undistort/rectify maps efficiently.

    Data flow (single frame)::

        FramePair
          → IRectifier.rectify()        → RectifiedPair
          → IDisparityMatcher.compute() → disparity (H×W float32)
          → IDepthEstimator.to_depth()  → DepthMap (left_rect=None)
          → attach RectifiedPair.left/right as DepthMap.left_rect/right_rect
          → IPostProcessor.process()    → DepthMap  (applied in order, optional)
          → return DepthMap

    Data flow (streaming)::

        ICameraSource.stream()
          └─▶ FramePair (repeated)
                └─▶ process()  → DepthMap
                      └─▶ yield DepthMap
    """

    def __init__(
        self,
        rectifier: IRectifier,
        matcher: IDisparityMatcher,
        depth_estimator: IDepthEstimator,
        calib: CalibrationResult,
        camera_source: Optional[ICameraSource] = None,
        post_processors: Optional[list[IPostProcessor]] = None,
    ) -> None:
        self._rectifier       = rectifier
        self._matcher         = matcher
        self._depth_estimator = depth_estimator
        self._calib           = calib
        self._camera_source   = camera_source
        self._post_processors: list[IPostProcessor] = post_processors or []

    def process(self, pair: FramePair) -> DepthMap:
        rect = self._rectifier.rectify(pair, self._calib)
        disp = self._matcher.compute(rect.left, rect.right)
        depth_map = self._depth_estimator.to_depth(disp, self._calib)
        result = DepthMap(
            data=depth_map.data,
            disparity=depth_map.disparity,
            left_rect=rect.left,
            right_rect=rect.right,
        )
        for pp in self._post_processors:
            result = pp.process(result)
        return result

    def stream(self) -> Iterator[DepthMap]:
        """Yield a DepthMap for every frame produced by the attached camera_source.

        Raises:
            RuntimeError: if no ``camera_source`` was provided at construction.
        """
        if self._camera_source is None:
            raise RuntimeError(
                "StereoPipeline.stream() requires a camera_source. "
                "Pass camera_source= to the constructor."
            )
        for pair in self._camera_source.stream():
            yield self.process(pair)
