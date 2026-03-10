from __future__ import annotations

from typing import Iterator, Optional, Union

import cv2
import numpy as np

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
        self._last_left_rect: Optional[np.ndarray] = None

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

    def process_stack(
        self,
        frames: Union[list[np.ndarray], np.ndarray],
        spatial_kernel: int = 5,
        temporal_frames: int = 15,
        min_depth_m: float = 0.10,
    ) -> np.ndarray:
        """Process a temporal stack of raw side-by-side frames into a stable depth map.

        Each frame must be a BGR SBS image with shape ``(H, W*2, 3)`` where
        ``(W, H) = calib.image_size``.

        Steps:

        1. Split each SBS frame into a ``FramePair`` and run the per-frame
           pipeline (rectify → disparity → depth → post-processors).
        2. Apply a ``spatial_kernel × spatial_kernel`` median filter to each
           per-frame depth map.
        3. Take a pixel-wise temporal median across all *N* depth frames.
        4. Mask pixels with depth < ``min_depth_m`` as ``NaN``.
        5. Return a ``(H, W)`` float32 array in metres.

        The rectified left image from the last frame is cached and returned by
        :meth:`last_left_rect`.

        Args:
            frames:          List (or array) of *N* raw SBS BGR frames.
            spatial_kernel:  Odd integer ≥ 3 for the spatial median kernel size.
            temporal_frames: Expected number of frames (validates ``len(frames)``).
            min_depth_m:     Pixels with depth below this value are set to ``NaN``.

        Returns:
            ``float32`` numpy array of shape ``(H, W)`` in metres.

        Raises:
            ValueError: if ``len(frames) != temporal_frames``, if
                ``spatial_kernel`` is even or < 3, or if any frame has an
                unexpected shape.
        """
        if isinstance(frames, np.ndarray):
            frames = list(frames)

        if len(frames) != temporal_frames:
            raise ValueError(
                f"Expected {temporal_frames} frames, got {len(frames)}."
            )
        if spatial_kernel < 3 or spatial_kernel % 2 == 0:
            raise ValueError(
                f"spatial_kernel must be an odd integer >= 3, got {spatial_kernel}."
            )

        w, h = self._calib.image_size
        expected_shape = (h, w * 2, 3)
        for i, frame in enumerate(frames):
            if frame.shape != expected_shape:
                raise ValueError(
                    f"Frame {i} has shape {frame.shape}; "
                    f"expected {expected_shape} "
                    f"(H={h}, W*2={w * 2}, 3 channels) "
                    f"from calib.image_size=({w}, {h})."
                )

        depth_stack: list[np.ndarray] = []
        last_left_rect: Optional[np.ndarray] = None

        for frame in frames:
            mid = frame.shape[1] // 2
            pair = FramePair(left=frame[:, :mid], right=frame[:, mid:])
            depth_map = self.process(pair)

            # Spatial median filter — cv2.medianBlur requires no NaN values.
            # Replace NaN with 0.0 before filtering; the final min_depth_m
            # threshold (step 4) will mask those pixels as NaN.
            depth_data = np.nan_to_num(depth_map.data, nan=0.0)
            filtered = cv2.medianBlur(depth_data, spatial_kernel)
            depth_stack.append(filtered)

            last_left_rect = depth_map.left_rect

        self._last_left_rect = last_left_rect

        # Temporal median across the stack (NaN-safe; pixels fixed at 0.0 are
        # treated as valid here but will be removed by the min_depth_m mask).
        stack = np.stack(depth_stack, axis=0)           # (N, H, W)
        temporal = np.nanmedian(stack, axis=0).astype(np.float32)

        temporal[temporal < min_depth_m] = np.nan
        return temporal

    def last_left_rect(self) -> np.ndarray:
        """Return the rectified left image from the most recent :meth:`process_stack` call.

        Raises:
            RuntimeError: if :meth:`process_stack` has not been called yet.
        """
        if self._last_left_rect is None:
            raise RuntimeError(
                "last_left_rect() called before process_stack(). "
                "Call process_stack() at least once first."
            )
        return self._last_left_rect

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
