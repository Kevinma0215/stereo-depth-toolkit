from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2

from stereo_depth.config.io import load_yaml
from stereo_depth.calib.rectify import build_rectify_maps, rectify_pair
from stereo_depth.depth.presets import preset
from stereo_depth.depth.sgbm import create_matcher, compute_disparity, disparity_to_depth

def _ensure_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def _vis_disparity(disp: np.ndarray) -> np.ndarray:
    d = disp.copy()
    d[np.isnan(d)] = 0
    d[d < 0] = 0
    # normalize for visualization
    v = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    v = cv2.applyColorMap(v, cv2.COLORMAP_JET)
    return v

def run_depth_once(
    calib_yaml: Path,
    left_path: Path,
    right_path: Path,
    out_dir: Path,
    *,
    preset_name: str = "indoor",
    save_npy: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = load_yaml(calib_yaml)
    maps = build_rectify_maps(calib)
    Q = np.array(calib["Q"], dtype=np.float64)

    L = cv2.imread(str(left_path))
    R = cv2.imread(str(right_path))
    if L is None or R is None:
        raise RuntimeError("Failed to read left/right images")

    Lr, Rr = rectify_pair(L, R, maps)
    cv2.imwrite(str(out_dir / "left_rect.png"), Lr)
    cv2.imwrite(str(out_dir / "right_rect.png"), Rr)

    Lg = _ensure_gray(Lr)
    Rg = _ensure_gray(Rr)

    p = preset(preset_name)
    matcher = create_matcher(p)

    disp = compute_disparity(Lg, Rg, matcher)
    depth = disparity_to_depth(disp, Q)

    disp_vis = _vis_disparity(disp)
    cv2.imwrite(str(out_dir / "disparity.png"), disp_vis)

    if save_npy:
        np.save(out_dir / "disparity.npy", disp)
        np.save(out_dir / "depth_m.npy", depth)

    return out_dir
