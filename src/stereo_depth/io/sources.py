from __future__ import annotations
import cv2

def open_source(*, device: int = 0, path: str | None = None, video: str | None = None,
                width: int = 0, height: int = 0, fps: int = 0) -> cv2.VideoCapture:
    if video:
        cap = cv2.VideoCapture(video)
    elif path:
        cap = cv2.VideoCapture(path)   # V4L2 path
    else:
        cap = cv2.VideoCapture(device)

    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError("Failed to open source")
    return cap
