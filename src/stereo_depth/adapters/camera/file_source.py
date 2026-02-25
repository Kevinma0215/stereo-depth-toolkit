from __future__ import annotations
from pathlib import Path
import cv2

from stereo_depth.use_cases.ports import ICameraSource
from stereo_depth.entities import FramePair


class FileSource(ICameraSource):
    """ICameraSource that loads a single stereo pair from two image files.

    Useful for offline processing and tests (no camera required).
    Calling ``grab()`` repeatedly always returns the same pair.
    """

    def __init__(self, left_path: str | Path, right_path: str | Path) -> None:
        self._left_path = Path(left_path)
        self._right_path = Path(right_path)

    def grab(self) -> FramePair:
        left = cv2.imread(str(self._left_path))
        right = cv2.imread(str(self._right_path))
        if left is None:
            raise RuntimeError(f"Failed to load left image: {self._left_path}")
        if right is None:
            raise RuntimeError(f"Failed to load right image: {self._right_path}")
        return FramePair(left=left, right=right)
