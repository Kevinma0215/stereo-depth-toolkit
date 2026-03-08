"""SGBM preset loader.

Preset parameters live in config/sgbm.yaml at the package root.
To tune or add presets, edit that file — no Python changes required.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from stereo_depth.infrastructure.config.io import load_yaml

# Resolve config path relative to the package root (src/stereo_depth/),
# two levels up from this file (adapters/matcher/).
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "sgbm.yaml"


@dataclass(frozen=True)
class SGBMPreset:
    min_disparity: int
    num_disparities: int   # must be divisible by 16
    block_size: int
    p1: int
    p2: int
    disp12_max_diff: int
    pre_filter_cap: int
    uniqueness_ratio: int
    speckle_window_size: int
    speckle_range: int
    mode: str  # "SGBM" | "HH"


def preset(name: str) -> SGBMPreset:
    name = name.lower()
    data = load_yaml(_CONFIG_PATH)
    if name not in data:
        available = ", ".join(sorted(data.keys()))
        raise ValueError(f"Unknown preset {name!r}. Available: {available}")
    return SGBMPreset(**data[name])
