from __future__ import annotations

from stereo_depth.entities import DepthMap
from stereo_depth.use_cases.ports import IPostProcessor


class IdentityPostProcessor(IPostProcessor):
    """IPostProcessor that returns the DepthMap unchanged.

    Useful as a no-op placeholder, for testing, or as the head of a
    composite post-processor chain.
    """

    def process(self, depth_map: DepthMap) -> DepthMap:
        return depth_map
