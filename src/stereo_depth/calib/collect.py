""" 
處理「從影像路徑收集」+ 統計 report
 """
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .detect import detect_charuco, CharucoDetection

@dataclass
class CollectReport:
    total: int
    ok: int
    fail_no_markers: int
    fail_too_few_markers: int
    fail_too_few_charuco: int
    min_charuco_seen: int
    max_charuco_seen: int

def collect_charuco_from_paths(
    img_paths: list[Path],
    board,
    dictionary,
    *,
    min_markers: int = 4,
    min_charuco: int = 10,
):
    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None

    stats = {
        "total": 0, "ok": 0,
        "no_markers": 0, "too_few_markers": 0, "too_few_charuco": 0,
        "min_charuco": 10**9, "max_charuco": 0
    }

    for p in img_paths:
        stats["total"] += 1
        img = cv2.imread(str(p))
        if img is None:
            stats["no_markers"] += 1
            continue

        if img_size is None:
            h, w = img.shape[:2]
            img_size = (w, h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        det: CharucoDetection = detect_charuco(gray, board, dictionary, min_markers=min_markers, min_charuco=min_charuco)

        stats["min_charuco"] = min(stats["min_charuco"], det.num_charuco)
        stats["max_charuco"] = max(stats["max_charuco"], det.num_charuco)

        if not det.ok:
            stats[det.reason or "no_markers"] += 1
            continue

        all_corners.append(det.corners)  # type: ignore[arg-type]
        all_ids.append(det.ids)          # type: ignore[arg-type]
        stats["ok"] += 1

    if img_size is None:
        img_size = (0, 0)

    report = CollectReport(
        total=stats["total"],
        ok=stats["ok"],
        fail_no_markers=stats["no_markers"],
        fail_too_few_markers=stats["too_few_markers"],
        fail_too_few_charuco=stats["too_few_charuco"],
        min_charuco_seen=(0 if stats["min_charuco"] == 10**9 else stats["min_charuco"]),
        max_charuco_seen=stats["max_charuco"],
    )
    return all_corners, all_ids, img_size, report
