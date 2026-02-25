from __future__ import annotations
from pathlib import Path


def list_images(folder: Path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    paths = []
    for e in exts:
        paths += list(folder.glob(e))
    return sorted(paths)


def pair_images(left_dir: Path, right_dir: Path, *, mode: str = "auto"):
    """Match left and right image files into ordered pairs.

    mode:
      - "name":  match by identical filename
      - "index": match by sorted order (0..min-1)
      - "auto":  try name, fallback to index
    """
    left = list_images(left_dir)
    right = list_images(right_dir)

    if len(left) == 0 or len(right) == 0:
        return [], left, right, "none"

    if mode in ("name", "auto"):
        r_map = {p.name: p for p in right}
        pairs_name = [(lp, r_map[lp.name]) for lp in left if lp.name in r_map]
        if mode == "name":
            return pairs_name, left, right, "name"
        if len(pairs_name) > 0:
            return pairs_name, left, right, "name"

    if mode in ("index", "auto"):
        n = min(len(left), len(right))
        pairs_idx = list(zip(left[:n], right[:n]))
        return pairs_idx, left, right, "index"

    return [], left, right, "none"
