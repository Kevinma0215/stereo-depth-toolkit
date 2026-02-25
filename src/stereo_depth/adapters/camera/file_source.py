from __future__ import annotations
from pathlib import Path
from typing import Iterator
import cv2

from stereo_depth.use_cases.ports import ICameraSource
from stereo_depth.entities import FramePair


class FileSource(ICameraSource):
    """ICameraSource that reads stereo pairs from files on disk.

    Single-pair mode (both paths are image files):
        ``grab()`` always returns the same pair.
        ``stream()`` yields that one pair once.

    Directory mode (both paths are directories):
        Pairs are matched by sorted filename across the two directories.
        ``grab()`` returns the first pair.
        ``stream()`` yields all pairs in sorted order.

    Useful for offline processing and tests (no camera required).
    """

    def __init__(self, left: str | Path, right: str | Path) -> None:
        self._left = Path(left)
        self._right = Path(right)

    def _load_pair(self, left_path: Path, right_path: Path) -> FramePair:
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None:
            raise RuntimeError(f"Failed to load left image: {left_path}")
        if right is None:
            raise RuntimeError(f"Failed to load right image: {right_path}")
        return FramePair(left=left, right=right)

    def grab(self) -> FramePair:
        if self._left.is_dir():
            left_files = sorted(p for p in self._left.iterdir() if p.is_file())
            right_files = sorted(p for p in self._right.iterdir() if p.is_file())
            if not left_files:
                raise RuntimeError(f"No images found in {self._left}")
            return self._load_pair(left_files[0], right_files[0])
        return self._load_pair(self._left, self._right)

    def stream(self) -> Iterator[FramePair]:
        if self._left.is_dir():
            left_files = sorted(p for p in self._left.iterdir() if p.is_file())
            right_files = sorted(p for p in self._right.iterdir() if p.is_file())
            for lf, rf in zip(left_files, right_files):
                yield self._load_pair(lf, rf)
        else:
            yield self._load_pair(self._left, self._right)
