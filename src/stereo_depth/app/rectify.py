from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

from stereo_depth.infrastructure.config.io import load_yaml
from stereo_depth.adapters.calibration.charuco_calibrator import build_rectify_maps, rectify_pair
from stereo_depth.infrastructure.io.pairs import pair_images

def _draw_epipolar_lines(img: np.ndarray, step: int = 40):
    out = img.copy()
    h = out.shape[0]
    for y in range(0, h, step):
        cv2.line(out, (0, y), (out.shape[1] - 1, y), (0, 255, 0), 1)
    return out

def run_rectify_dataset(
    calib_yaml: Path,
    data_dir: Path,
    out_dir: Path,
    *,
    limit: int = 0,
    preview: bool = False,
    pairing: str = "auto"
):
    calib = load_yaml(calib_yaml)
    maps = build_rectify_maps(calib)

    left_dir = data_dir / "left"
    right_dir = data_dir / "right"
    pairs, left_all, right_all, mode_used = pair_images(left_dir, right_dir, mode=pairing)
    if len(pairs) == 0:
        sample_left = [p.name for p in left_all[:5]]
        sample_right = [p.name for p in right_all[:5]]
        raise RuntimeError(
            "No paired images found.\n"
            f"left_dir={left_dir} (n={len(left_all)}) sample={sample_left}\n"
            f"right_dir={right_dir} (n={len(right_all)}) sample={sample_right}\n"
            "Tip: ensure filenames match, or use index pairing."
        )


    if limit and limit > 0:
        pairs = pairs[:limit]

    out_left = out_dir / "left"
    out_right = out_dir / "right"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)

    print(f"[rectify] pairing mode used: {mode_used}, pairs={len(pairs)}")

    # optional preview image (save to disk, no GUI needed)
    preview_path = out_dir / "rectify_preview.png"

    for i, (lp, rp) in enumerate(pairs):
        L = cv2.imread(str(lp))
        R = cv2.imread(str(rp))
        if L is None or R is None:
            continue

        Lr, Rr = rectify_pair(L, R, maps)

        cv2.imwrite(str(out_left / lp.name), Lr)
        cv2.imwrite(str(out_right / rp.name), Rr)

        if preview and i == 0:
            # make a side-by-side preview with epipolar lines
            Lg = _draw_epipolar_lines(Lr, step=40)
            Rg = _draw_epipolar_lines(Rr, step=40)
            vis = cv2.hconcat([Lg, Rg])
            cv2.imwrite(str(preview_path), vis)

    return out_dir, (preview_path if preview else None), len(pairs), mode_used
