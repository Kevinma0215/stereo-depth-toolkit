"""YAML persistence for single-camera intrinsics.

Deliberately a separate file format from the stereo ``calib.yaml``: the two
describe different things, and a schema tag lets each reader reject the other
with a clear message instead of a bare KeyError from deep inside.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from stereo_depth.entities import MonoIntrinsics
from stereo_depth.infrastructure.config.io import load_yaml, save_yaml
from stereo_depth.use_cases.ports import IMonoCalibrationRepo

SCHEMA = "stereo_depth/mono_intrinsics@1"

PAIRING_NOTE = (
    "K/D describe RAW (distorted) images. K_new describes UNDISTORTED images "
    "and carries ZERO distortion. Never pair K with undistorted frames or "
    "K_new with raw frames."
)


class MonoYamlCalibrationRepo(IMonoCalibrationRepo):
    """Persists MonoIntrinsics as YAML.

    ``save`` writes only the core intrinsics; ``app/calibrate_mono.py`` layers
    the model-comparison table, undistortion matrices and Isaac Sim blocks on
    top of the same file via :func:`build_document`.
    """

    def load(self, path: str) -> MonoIntrinsics:
        data = load_yaml(Path(path))
        schema = data.get("schema")
        if schema != SCHEMA:
            raise ValueError(
                f"{path} is not a mono calibration file "
                f"(expected schema {SCHEMA!r}, found {schema!r}). "
                "Stereo calibrations are loaded with YamlCalibrationRepo."
            )
        return MonoIntrinsics(
            model=str(data["model"]),
            image_size=(int(data["image_size"]["width"]),
                        int(data["image_size"]["height"])),
            K=np.array(data["K"], dtype=np.float64),
            D=np.array(data["D"], dtype=np.float64),
            rpe_px=float(data.get("rpe_px", 0.0)),
            views_used=int(data.get("views_used", 0)),
        )

    def save(self, result: MonoIntrinsics, path: str) -> None:
        save_yaml(Path(path), core_document(result))

    def load_raw(self, path: str) -> dict:
        """Full document, including the sections ``load`` does not model."""
        data = load_yaml(Path(path))
        schema = data.get("schema")
        if schema != SCHEMA:
            raise ValueError(
                f"{path} is not a mono calibration file "
                f"(expected schema {SCHEMA!r}, found {schema!r})."
            )
        return data


def core_document(result: MonoIntrinsics) -> dict:
    """The minimal, always-present part of a mono calibration document."""
    return {
        "schema": SCHEMA,
        "model": result.model,
        "image_size": {
            "width": int(result.image_size[0]),
            "height": int(result.image_size[1]),
        },
        "K": np.asarray(result.K, dtype=float).tolist(),
        "D": np.asarray(result.D, dtype=float).ravel().tolist(),
        "rpe_px": float(result.rpe_px),
        "views_used": int(result.views_used),
        "note": PAIRING_NOTE,
    }


def build_document(
    result: MonoIntrinsics,
    *,
    board: dict | None = None,
    selection: dict | None = None,
    models: dict | None = None,
    undistort: dict | None = None,
    isaac_sim: dict | None = None,
) -> dict:
    """Assemble the full mono calibration document in a stable key order."""
    doc = core_document(result)
    for key, value in (
        ("board", board),
        ("selection", selection),
        ("models", models),
        ("undistort", undistort),
        ("isaac_sim", isaac_sim),
    ):
        if value is not None:
            doc[key] = value
    return doc
