from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2

from stereo_depth.adapters.calibration.yaml_repo import YamlCalibrationRepo
from stereo_depth.adapters.camera.file_source import FileSource
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.adapters.depth.opencv_depth_estimator import OpenCVDepthEstimator
from stereo_depth.use_cases.pipeline import StereoPipeline
from stereo_depth.use_cases.ports import IDisparityMatcher


def _vis_disparity(disp: np.ndarray) -> np.ndarray:
    d = disp.copy()
    d[np.isnan(d)] = 0
    d[d < 0] = 0
    v = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    v = cv2.applyColorMap(v, cv2.COLORMAP_JET)
    return v


def _build_matcher(matcher_name: str, preset_name: str, image_size: tuple[int, int]) -> IDisparityMatcher:
    if matcher_name == "sgbm":
        return SgbmMatcher(preset_name=preset_name)
    if matcher_name == "retinify":
        # Lazy import: retinify_matcher raises ImportError at module level when
        # the retinify package is absent, so we defer the import to call time.
        from stereo_depth.adapters.matcher.retinify_matcher import RetinifyMatcher  # noqa: PLC0415
        w, h = image_size
        return RetinifyMatcher(width=w, height=h, mode=preset_name)
    raise ValueError(
        f"Unknown matcher '{matcher_name}'. Valid options: sgbm | retinify"
    )


def run_depth_once(
    calib_yaml: Path,
    left_path: Path,
    right_path: Path,
    out_dir: Path,
    *,
    preset_name: str = "indoor",
    save_npy: bool = True,
    matcher_name: str = "sgbm",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = YamlCalibrationRepo().load(str(calib_yaml))
    matcher = _build_matcher(matcher_name, preset_name, calib.image_size)

    pipeline = StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=matcher,
        depth_estimator=OpenCVDepthEstimator(),
        calib=calib,
    )

    pair = FileSource(left_path, right_path).grab()
    depth_map = pipeline.process(pair)

    cv2.imwrite(str(out_dir / "left_rect.png"), depth_map.left_rect)
    cv2.imwrite(str(out_dir / "right_rect.png"), depth_map.right_rect)
    cv2.imwrite(str(out_dir / "disparity.png"), _vis_disparity(depth_map.disparity))

    if save_npy:
        np.save(out_dir / "disparity.npy", depth_map.disparity)
        np.save(out_dir / "depth_m.npy", depth_map.data)

    return out_dir
