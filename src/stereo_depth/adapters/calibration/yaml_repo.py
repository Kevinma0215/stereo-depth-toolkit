from __future__ import annotations
from pathlib import Path
import numpy as np

from stereo_depth.use_cases.ports import ICalibrationRepo
from stereo_depth.entities import CalibrationResult
from stereo_depth.infrastructure.config.io import load_yaml, save_yaml


class YamlCalibrationRepo(ICalibrationRepo):
    """ICalibrationRepo that persists CalibrationResult as YAML.

    The YAML schema mirrors the one produced by ``app/calibrate.py``:

    .. code-block:: yaml

        image_size: {width: W, height: H}
        K1: [[...], [...], [...]]
        D1: [...]
        K2/D2/R/T/baseline_m/R1/R2/P1/P2/Q: ...
        metrics: {stereo_rms: ..., ...}   # optional; rpe_px preferred

    ``rpe_px`` may be absent in existing files; falls back to
    ``metrics.stereo_rms`` if available, else 0.0.
    """

    def load(self, path: str) -> CalibrationResult:
        data = load_yaml(path)
        w = int(data["image_size"]["width"])
        h = int(data["image_size"]["height"])
        rpe = float(
            data.get("rpe_px", data.get("metrics", {}).get("stereo_rms", 0.0))
        )
        return CalibrationResult(
            image_size=(w, h),
            K1=np.array(data["K1"], dtype=np.float64),
            D1=np.array(data["D1"], dtype=np.float64),
            K2=np.array(data["K2"], dtype=np.float64),
            D2=np.array(data["D2"], dtype=np.float64),
            R=np.array(data["R"], dtype=np.float64),
            T=np.array(data["T"], dtype=np.float64),
            baseline_m=float(data["baseline_m"]),
            R1=np.array(data["R1"], dtype=np.float64),
            R2=np.array(data["R2"], dtype=np.float64),
            P1=np.array(data["P1"], dtype=np.float64),
            P2=np.array(data["P2"], dtype=np.float64),
            Q=np.array(data["Q"], dtype=np.float64),
            rpe_px=rpe,
        )

    def save(self, result: CalibrationResult, path: str) -> None:
        data = {
            "image_size": {
                "width": result.image_size[0],
                "height": result.image_size[1],
            },
            "K1": result.K1.tolist(),
            "D1": result.D1.tolist(),
            "K2": result.K2.tolist(),
            "D2": result.D2.tolist(),
            "R": result.R.tolist(),
            "T": result.T.tolist(),
            "baseline_m": result.baseline_m,
            "R1": result.R1.tolist(),
            "R2": result.R2.tolist(),
            "P1": result.P1.tolist(),
            "P2": result.P2.tolist(),
            "Q": result.Q.tolist(),
            "rpe_px": result.rpe_px,
        }
        save_yaml(Path(path), data)
