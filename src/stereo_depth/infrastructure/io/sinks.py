from __future__ import annotations
from pathlib import Path
import cv2


class VideoRecorder:
    def __init__(self, out_path: Path, fourcc: str = "MJPG"):
        self.out_path = out_path
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer: cv2.VideoWriter | None = None

    def maybe_init(self, frame, fps: float):
        if self.writer is not None:
            return
        h, w = frame.shape[:2]
        if fps <= 0:
            fps = 30
        self.writer = cv2.VideoWriter(str(self.out_path), self.fourcc, fps, (w, h))

    def write(self, frame):
        if self.writer is None:
            raise RuntimeError("Recorder not initialized. Call maybe_init first.")
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
