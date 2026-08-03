"""Quality gates for live auto-collection of calibration views.

Pure logic — no camera, no windows — so it is fully unit-testable. The live
loop in ``app/calibrate_mono.py`` feeds each frame's ChArUco detection in and
renders the returned :class:`GateStatus`.

Four gates, each guarding a specific way a calibration goes wrong:

* **Coverage** — distortion coefficients are driven by corners far from the
  image centre. A session shot entirely in the middle of the frame produces
  confident-looking but badly extrapolated coefficients, especially on a
  wide-angle lens.
* **Sharpness / steadiness** — motion blur biases corner localisation.
* **Corner count** — too few corners makes a view's pose ill-conditioned.
* **Tilt diversity** — all-frontal views leave fx/fy and the radial terms
  mutually unidentifiable; the board must be seen at an angle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from stereo_depth.adapters.calibration.charuco_calibrator import CharucoDetection

# Tilt bins. "frontal" alone is degenerate, hence the requirement to fill
# several of these before a session counts as complete.
TILT_BINS = ("frontal", "left", "right", "up", "down")

_CELL_LABELS = {
    (0, 0): "TOP-LEFT",    (0, 1): "TOP",    (0, 2): "TOP-RIGHT",
    (1, 0): "LEFT",        (1, 1): "CENTRE", (1, 2): "RIGHT",
    (2, 0): "BOTTOM-LEFT", (2, 1): "BOTTOM", (2, 2): "BOTTOM-RIGHT",
}


def cell_label(row: int, col: int, rows: int = 3, cols: int = 3) -> str:
    """Human-readable name for a coverage cell."""
    if (rows, cols) == (3, 3):
        return _CELL_LABELS[(row, col)]
    return f"R{row + 1}C{col + 1}"


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

class CoverageTracker:
    """Tracks which regions of the frame the board's corners have visited.

    A cell counts as visited when it holds at least ``min_corners`` ChArUco
    corners — corners, not the board centre, because it is the corners that
    actually constrain the distortion model.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        rows: int = 3,
        cols: int = 3,
        *,
        min_corners: int = 6,
    ) -> None:
        self.image_size = image_size
        self.rows = rows
        self.cols = cols
        self.min_corners = min_corners
        self.covered: set[tuple[int, int]] = set()

    @property
    def total_cells(self) -> int:
        return self.rows * self.cols

    def cells_of(self, corners: np.ndarray | None) -> set[tuple[int, int]]:
        """Cells holding at least ``min_corners`` of the given corners."""
        if corners is None or len(corners) == 0:
            return set()
        w, h = self.image_size
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        col = np.clip((pts[:, 0] / w * self.cols).astype(int), 0, self.cols - 1)
        row = np.clip((pts[:, 1] / h * self.rows).astype(int), 0, self.rows - 1)

        counts: dict[tuple[int, int], int] = {}
        for r, c in zip(row, col):
            key = (int(r), int(c))
            counts[key] = counts.get(key, 0) + 1
        return {k for k, n in counts.items() if n >= self.min_corners}

    def commit(self, corners: np.ndarray | None) -> set[tuple[int, int]]:
        """Record a view; returns the cells it covered for the first time."""
        cells = self.cells_of(corners)
        new = cells - self.covered
        self.covered |= cells
        return new

    def missing(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.covered
        ]

    def nearest_missing_label(self, corners: np.ndarray | None) -> str | None:
        """Label of the uncovered cell closest to where the board is now."""
        missing = self.missing()
        if not missing:
            return None
        if corners is None or len(corners) == 0:
            r, c = missing[0]
            return cell_label(r, c, self.rows, self.cols)

        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        w, h = self.image_size
        cx = pts[:, 0].mean() / w * self.cols - 0.5
        cy = pts[:, 1].mean() / h * self.rows - 0.5
        r, c = min(missing, key=lambda rc: (rc[0] - cy) ** 2 + (rc[1] - cx) ** 2)
        return cell_label(r, c, self.rows, self.cols)


# ---------------------------------------------------------------------------
# Sharpness / steadiness
# ---------------------------------------------------------------------------

def blur_score(gray: np.ndarray, corners: np.ndarray | None = None) -> float:
    """Laplacian variance over the board's bounding box.

    Restricting to the board makes the score independent of whatever else is
    in the scene — a busy background would otherwise mask a blurred board.
    """
    roi = gray
    if corners is not None and len(corners) > 0:
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        h, w = gray.shape[:2]
        x0 = int(max(0, np.floor(pts[:, 0].min())))
        x1 = int(min(w, np.ceil(pts[:, 0].max()) + 1))
        y0 = int(max(0, np.floor(pts[:, 1].min())))
        y1 = int(min(h, np.ceil(pts[:, 1].max()) + 1))
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            roi = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


class SteadyTracker:
    """Board is steady when it has barely moved for a few frames running.

    Between-frame motion is a good proxy for during-exposure motion, and it
    catches the smooth drift that a Laplacian check alone will happily pass.
    """

    def __init__(self, *, max_motion_px: float = 2.0, frames: int = 4) -> None:
        self.max_motion_px = max_motion_px
        self.frames = frames
        self._prev_centroid: np.ndarray | None = None
        self._streak = 0
        self.last_motion_px: float = float("inf")

    def reset(self) -> None:
        self._prev_centroid = None
        self._streak = 0
        self.last_motion_px = float("inf")

    def update(self, corners: np.ndarray | None) -> bool:
        if corners is None or len(corners) == 0:
            self.reset()
            return False
        centroid = np.asarray(corners, dtype=np.float64).reshape(-1, 2).mean(axis=0)
        if self._prev_centroid is None:
            self._prev_centroid = centroid
            self._streak = 0
            self.last_motion_px = float("inf")
            return False

        motion = float(np.linalg.norm(centroid - self._prev_centroid))
        self._prev_centroid = centroid
        self.last_motion_px = motion
        self._streak = self._streak + 1 if motion <= self.max_motion_px else 0
        return self._streak >= self.frames

    @property
    def is_steady(self) -> bool:
        return self._streak >= self.frames


# ---------------------------------------------------------------------------
# Tilt diversity
# ---------------------------------------------------------------------------

class TiltTracker:
    """Classifies board pose into coarse tilt bins via solvePnP.

    The intrinsic guess only has to be good enough to sort poses into bins,
    so a crude fx = 0.8 * width is fine — no calibration exists yet.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        frontal_max_deg: float = 10.0,
        tilt_max_deg: float = 50.0,
    ) -> None:
        w, h = image_size
        self.K_guess = np.array([[0.8 * w, 0.0, w / 2.0],
                                 [0.0, 0.8 * w, h / 2.0],
                                 [0.0, 0.0, 1.0]])
        self.frontal_max_deg = frontal_max_deg
        self.tilt_max_deg = tilt_max_deg
        self.filled: set[str] = set()

    def pose_of(
        self, corners: np.ndarray | None, ids: np.ndarray | None, board
    ) -> tuple[float, float] | None:
        """(tilt_deg, azimuth_deg) of the board plane, or None."""
        if corners is None or ids is None or len(corners) < 6:
            return None
        obj, img = board.matchImagePoints(corners, ids)
        if obj is None or len(obj) < 6:
            return None
        ok, rvec, _tvec = cv2.solvePnP(
            obj.astype(np.float64), img.astype(np.float64),
            self.K_guess, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        normal = R @ np.array([0.0, 0.0, 1.0])
        # angle between the board normal and the camera's viewing axis
        tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
        azimuth = math.degrees(math.atan2(float(normal[1]), float(normal[0])))
        return tilt, azimuth

    def bin_of(self, tilt_deg: float, azimuth_deg: float) -> str | None:
        """Tilt bin name, or None when the board is tilted too far to trust."""
        if tilt_deg <= self.frontal_max_deg:
            return "frontal"
        if tilt_deg > self.tilt_max_deg:
            return None          # extreme foreshortening localises corners badly
        az = azimuth_deg % 360.0
        if az < 45.0 or az >= 315.0:
            return "right"
        if az < 135.0:
            return "down"
        if az < 225.0:
            return "left"
        return "up"

    def commit(self, bin_name: str | None) -> bool:
        """Record a bin; True when it had not been seen before."""
        if bin_name is None or bin_name in self.filled:
            return False
        self.filled.add(bin_name)
        return True

    def missing(self) -> list[str]:
        return [b for b in TILT_BINS if b not in self.filled]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class GateStatus:
    """Per-frame verdict; also everything the HUD needs to render."""
    detect_ok: bool = False
    sharp_ok: bool = False
    steady_ok: bool = False
    novel_ok: bool = False
    cooldown_ok: bool = False
    guidance: str = ""
    blur: float = 0.0
    motion_px: float = float("inf")
    num_charuco: int = 0
    tilt_bin: str | None = None
    tilt_deg: float | None = None
    current_cells: set[tuple[int, int]] = field(default_factory=set)
    new_cells: set[tuple[int, int]] = field(default_factory=set)

    @property
    def all_ok(self) -> bool:
        return (
            self.detect_ok and self.sharp_ok and self.steady_ok
            and self.novel_ok and self.cooldown_ok
        )


class AutoCollectPolicy:
    """Decides, frame by frame, whether a view is worth keeping.

    Novelty is what stops the buffer filling with near-duplicate frames: a
    view is only accepted if it reaches an uncovered cell, fills a new tilt
    bin, or differs enough in position or apparent scale from the last one.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        board,
        *,
        target_views: int = 40,
        blur_min: float = 60.0,
        steady_px: float = 2.0,
        steady_frames: int = 4,
        cooldown_s: float = 0.7,
        rows: int = 3,
        cols: int = 3,
        min_cell_corners: int = 6,
        move_frac: float = 0.05,
        scale_frac: float = 0.15,
        min_tilt_bins: int = 4,
    ) -> None:
        self.image_size = image_size
        self.board = board
        self.target_views = target_views
        self.blur_min = blur_min
        self.cooldown_s = cooldown_s
        self.move_frac = move_frac
        self.scale_frac = scale_frac
        self.min_tilt_bins = min_tilt_bins

        self.coverage = CoverageTracker(image_size, rows, cols, min_corners=min_cell_corners)
        self.steady = SteadyTracker(max_motion_px=steady_px, frames=steady_frames)
        self.tilt = TiltTracker(image_size)

        self.accepted = 0
        self._last_accept_t: float | None = None
        self._last_centroid: np.ndarray | None = None
        self._last_extent: float | None = None
        self._diag = float(np.hypot(*image_size))

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _centroid_extent(corners: np.ndarray) -> tuple[np.ndarray, float]:
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        centroid = pts.mean(axis=0)
        extent = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
        return centroid, extent

    def _is_novel(self, corners: np.ndarray, new_cells: set, new_tilt: bool) -> bool:
        if new_cells or new_tilt:
            return True
        if self._last_centroid is None:
            return True
        centroid, extent = self._centroid_extent(corners)
        moved = float(np.linalg.norm(centroid - self._last_centroid))
        if moved >= self.move_frac * self._diag:
            return True
        if self._last_extent and self._last_extent > 1e-6:
            if abs(extent - self._last_extent) / self._last_extent >= self.scale_frac:
                return True
        return False

    # -- main API ---------------------------------------------------------

    def evaluate(self, det: CharucoDetection, gray: np.ndarray, now: float) -> GateStatus:
        """Score the current frame without changing any collection state."""
        st = GateStatus()

        if not det.ok or det.corners is None:
            self.steady.reset()
            st.guidance = {
                "no_markers": "No board found - point the camera at the board",
                "too_few_markers": "Move closer - too few markers visible",
                "too_few_charuco": "Move closer - too few corners detected",
            }.get(det.reason or "no_markers", "No board found")
            st.num_charuco = det.num_charuco
            return st

        st.detect_ok = True
        st.num_charuco = det.num_charuco

        st.blur = blur_score(gray, det.corners)
        st.sharp_ok = st.blur >= self.blur_min

        st.steady_ok = self.steady.update(det.corners)
        st.motion_px = self.steady.last_motion_px

        pose = self.tilt.pose_of(det.corners, det.ids, self.board)
        if pose is not None:
            st.tilt_deg = pose[0]
            st.tilt_bin = self.tilt.bin_of(*pose)

        st.current_cells = self.coverage.cells_of(det.corners)
        st.new_cells = st.current_cells - self.coverage.covered
        new_tilt = st.tilt_bin is not None and st.tilt_bin not in self.tilt.filled
        st.novel_ok = self._is_novel(det.corners, st.new_cells, new_tilt)

        st.cooldown_ok = (
            self._last_accept_t is None or (now - self._last_accept_t) >= self.cooldown_s
        )

        st.guidance = self._guidance(st, det.corners)
        return st

    def _guidance(self, st: GateStatus, corners: np.ndarray | None) -> str:
        """Single highest-priority instruction to show the user."""
        if not st.sharp_ok:
            return "Blurry - hold the board still"
        if not st.steady_ok:
            return "Hold still..."
        if not st.cooldown_ok:
            return "Captured - keep moving"
        if not st.novel_ok:
            target = self.coverage.nearest_missing_label(corners)
            if target:
                return f"Move the board to {target}"
            missing_tilts = self.tilt.missing()
            if missing_tilts:
                return f"Tilt the board {missing_tilts[0].upper()}"
            return "Move or rotate the board to a new position"
        if st.new_cells:
            return "Good - capturing"
        return "Good - capturing"

    def accept(self, det: CharucoDetection, now: float) -> None:
        """Commit a view: update coverage, tilt, cooldown and novelty state."""
        if det.corners is None:
            return
        self.coverage.commit(det.corners)
        pose = self.tilt.pose_of(det.corners, det.ids, self.board)
        if pose is not None:
            self.tilt.commit(self.tilt.bin_of(*pose))
        self._last_centroid, self._last_extent = self._centroid_extent(det.corners)
        self._last_accept_t = now
        self.accepted += 1
        self.steady.reset()

    def undo(self) -> None:
        """Step the counter back after a rejected save.

        Coverage and tilt history are deliberately kept: they record where the
        board has been, and re-opening them would let a duplicate straight in.
        """
        self.accepted = max(0, self.accepted - 1)
        self._last_accept_t = None

    def done(self) -> bool:
        return (
            self.accepted >= self.target_views
            and not self.coverage.missing()
            and len(self.tilt.filled) >= self.min_tilt_bins
        )

    def progress(self) -> dict:
        return {
            "views": self.accepted,
            "target_views": self.target_views,
            "cells": len(self.coverage.covered),
            "total_cells": self.coverage.total_cells,
            "tilts": len(self.tilt.filled),
            "min_tilt_bins": self.min_tilt_bins,
        }
