"""UVC / V4L2 camera source — canonical location (moved from io/sources.py)."""
from __future__ import annotations
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

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        self._cap.release()
