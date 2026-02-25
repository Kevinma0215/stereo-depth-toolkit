from __future__ import annotations
from dataclasses import dataclass

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
    if name == "indoor":
        # 室內：視差通常較大，雜訊多，稍微強 regularization
        return SGBMPreset(
            min_disparity=0,
            num_disparities=256,   # 16 的倍數
            block_size=7,
            p1=8 * 3 * 5**2,
            p2=32 * 3 * 5**2,
            disp12_max_diff=1,
            pre_filter_cap=31,
            uniqueness_ratio=15,
            speckle_window_size=200,
            speckle_range=3,
            mode="SGBM",
        )
    if name == "outdoor":
        return SGBMPreset(
            min_disparity=0,
            num_disparities=192,
            block_size=5,
            p1=8 * 3 * 5**2,
            p2=32 * 3 * 5**2,
            disp12_max_diff=1,
            pre_filter_cap=31,
            uniqueness_ratio=12,
            speckle_window_size=200,
            speckle_range=3,
            mode="SGBM",
        )
    if name == "high_quality":
        return SGBMPreset(
            min_disparity=0,
            num_disparities=160,
            block_size=7,
            p1=8 * 3 * 7**2,
            p2=32 * 3 * 7**2,
            disp12_max_diff=1,
            pre_filter_cap=31,
            uniqueness_ratio=8,
            speckle_window_size=150,
            speckle_range=2,
            mode="HH",  # 速度慢但更穩
        )
    raise ValueError(f"Unknown preset: {name}. Use indoor|outdoor|high_quality")
