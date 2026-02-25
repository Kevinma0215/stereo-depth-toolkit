"""UVC / V4L2 camera source — canonical location (moved from io/sources.py)."""
from __future__ import annotations
from typing import Iterator, Optional
import cv2

from stereo_depth.use_cases.ports import ICameraSource
from stereo_depth.entities import FramePair
from stereo_depth.infrastructure.io.sbs_capture import SBSSplitter


# ---------------------------------------------------------------------------
# Low-level helper (previously io/sources.py:open_source)
# ---------------------------------------------------------------------------

def open_source(
    *,
    device: int = 0,
    path: str | None = None,
    video: str | None = None,
    width: int = 0,
    height: int = 0,
    fps: int = 0,
) -> cv2.VideoCapture:
    """Open a VideoCapture from a device index, V4L2 path, or video file."""
    if video:
        cap = cv2.VideoCapture(video)
    elif path:
        cap = cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture(device)

    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError("Failed to open source")
    return cap


# ---------------------------------------------------------------------------
# ICameraSource implementation
# ---------------------------------------------------------------------------

class UVCSource(ICameraSource):
    """ICameraSource backed by a UVC/V4L2 SBS camera.

    Reads one SBS frame per ``grab()`` call, splits it with ``splitter``,
    and returns a ``FramePair``.

    Args:
        cap:      An already-opened ``cv2.VideoCapture`` (use ``open_source()``).
        splitter: ``SBSSplitter`` configured for swap/crop if needed.
    """

    def __init__(self, cap: cv2.VideoCapture, splitter: SBSSplitter) -> None:
        self._cap = cap
        self._splitter = splitter

    def grab(self) -> FramePair:
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from UVC camera")
        left, right = self._splitter.split(frame)
        return FramePair(left=left, right=right)

    def stream(self, max_frames: Optional[int] = None) -> Iterator[FramePair]:
        """Yield FramePairs until the capture ends or max_frames is reached.

        Guarantees ``cap.release()`` is called on exit regardless of how the
        iteration ends (normal exhaustion, break, or KeyboardInterrupt).
        """
        count = 0
        try:
            while self._cap.isOpened():
                if max_frames is not None and count >= max_frames:
                    break
                ok, frame = self._cap.read()
                if not ok:
                    break
                left, right = self._splitter.split(frame)
                yield FramePair(left=left, right=right)
                count += 1
        finally:
            self._cap.release()

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        self._cap.release()


# ---------------------------------------------------------------------------
# Convenience wrapper — simple API for the common SBS camera use-case
# ---------------------------------------------------------------------------

class UvcSource(ICameraSource):
    """Convenience ICameraSource for the SBS UVC camera.

    Wraps ``open_source`` + ``SBSSplitter`` so callers only need:

    .. code-block:: python

        source = UvcSource(device_index=0, width=2560, height=720)

    Args:
        device_index: V4L2 device index (default 0).
        width:        Requested frame width in pixels (0 = driver default).
        height:       Requested frame height in pixels (0 = driver default).
        fps:          Requested frames-per-second (0 = driver default).
        swap_lr:      Swap left/right halves of the SBS frame.
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        swap_lr: bool = False,
    ) -> None:
        cap = open_source(device=device_index, width=width, height=height, fps=fps)
        self._inner = UVCSource(cap, SBSSplitter(swap_lr=swap_lr))

    def grab(self) -> FramePair:
        return self._inner.grab()

    def stream(self, max_frames: Optional[int] = None) -> Iterator[FramePair]:
        return self._inner.stream(max_frames=max_frames)

    def release(self) -> None:
        self._inner.release()
