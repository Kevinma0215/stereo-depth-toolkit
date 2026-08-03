"""Shared OpenCV HUD drawing helpers.

All functions draw in-place on a BGR uint8 image.
"""
from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN = (0, 200, 0)
YELLOW = (0, 200, 255)
RED = (0, 0, 220)
GREY = (150, 150, 150)
WHITE = (220, 220, 220)


def status_color(ok: bool, warn: bool = False) -> tuple[int, int, int]:
    """Green when satisfied, yellow when close, red otherwise."""
    if ok:
        return GREEN
    return YELLOW if warn else RED


def draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = WHITE,
    thick: int = 1,
    shadow: bool = True,
) -> None:
    """Draw text with a dark outline so it stays readable on any background."""
    if shadow:
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_banner(
    img: np.ndarray,
    text: str,
    *,
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.9,
    y: int | None = None,
) -> None:
    """Centred instruction line over a translucent strip."""
    if not text:
        return
    h, w = img.shape[:2]
    thick = 2
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    cy = h // 2 if y is None else y
    x = max(0, (w - tw) // 2)

    pad = 10
    y0 = max(0, cy - th - pad)
    y1 = min(h, cy + base + pad)
    if y1 > y0:
        strip = img[y0:y1]
        strip[:] = cv2.addWeighted(strip, 0.45, np.zeros_like(strip), 0.55, 0)

    cv2.putText(img, text, (x, cy), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, cy), FONT, scale, color, thick, cv2.LINE_AA)


def draw_progress_bar(
    img: np.ndarray,
    fraction: float,
    *,
    color: tuple[int, int, int] = GREEN,
    height: int = 6,
) -> None:
    """Progress bar pinned to the bottom edge."""
    h, w = img.shape[:2]
    frac = float(np.clip(fraction, 0.0, 1.0))
    cv2.rectangle(img, (0, h - height), (w, h), (60, 60, 60), -1)
    cv2.rectangle(img, (0, h - height), (int(w * frac), h), color, -1)


def draw_status_line(
    img: np.ndarray,
    items: list[tuple[str, bool]],
    origin: tuple[int, int],
    *,
    scale: float = 0.5,
    gap: int = 14,
) -> None:
    """Row of green/red labelled status dots, e.g. a gate checklist."""
    x, y = origin
    for label, ok in items:
        cv2.circle(img, (x + 5, y - 4), 5, status_color(ok), -1)
        cv2.circle(img, (x + 5, y - 4), 5, (0, 0, 0), 1, cv2.LINE_AA)
        draw_text(img, label, (x + 16, y), scale=scale,
                  color=WHITE if ok else GREY, thick=1)
        (tw, _), _ = cv2.getTextSize(label, FONT, scale, 1)
        x += 16 + tw + gap


def draw_coverage_inset(
    img: np.ndarray,
    covered: set[tuple[int, int]],
    active: set[tuple[int, int]],
    rows: int,
    cols: int,
    *,
    origin: tuple[int, int] = (10, 10),
    cell_px: int = 30,
) -> None:
    """Miniature map of the frame showing which cells have been covered.

    Filled green = covered, yellow outline = where the board is right now,
    dim red = still missing.
    """
    ox, oy = origin
    for r in range(rows):
        for c in range(cols):
            x0 = ox + c * cell_px
            y0 = oy + r * cell_px
            x1, y1 = x0 + cell_px - 2, y0 + cell_px - 2
            if (r, c) in covered:
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 120, 0), -1)
            else:
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 70), -1)
            border = YELLOW if (r, c) in active else (90, 90, 90)
            cv2.rectangle(img, (x0, y0), (x1, y1), border,
                          2 if (r, c) in active else 1)
