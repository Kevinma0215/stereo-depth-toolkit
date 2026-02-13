from __future__ import annotations
import cv2
from dataclasses import dataclass

@dataclass
class SBSSplitter:
    """Split a side-by-side (SBS) frame into (left, right)."""
    swap_lr: bool = False
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0  # 0 means no crop
    crop_h: int = 0

    def split(self, frame):
        if frame is None:
            raise ValueError("Empty frame")
        h, w = frame.shape[:2]

        # Optional crop (useful if camera adds borders/overlays)
        x0, y0 = self.crop_x, self.crop_y
        x1 = w if self.crop_w <= 0 else min(w, x0 + self.crop_w)
        y1 = h if self.crop_h <= 0 else min(h, y0 + self.crop_h)
        frame = frame[y0:y1, x0:x1]
        h, w = frame.shape[:2]

        mid = w // 2
        left = frame[:, :mid]
        right = frame[:, mid:]

        if self.swap_lr:
            left, right = right, left
        return left, right


def open_camera(device: int, width: int = 0, height: int = 0, fps: int = 0):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device {device}")
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap
