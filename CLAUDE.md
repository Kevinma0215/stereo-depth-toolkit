# CLAUDE.md

## Project Goal

Production-quality Python library for real-time stereo depth estimation using the HBVCAM-W202011HD camera.

- Input: stereo image pair (left + right)
- Output: float32 depth map (metres); future: 3D point cloud
- Key constraint: swapping the disparity matcher (e.g. SGBM → Retinify) requires **zero changes** outside `adapters/matcher/`

## Environment

```bash
conda activate cv   # Python 3.10
pip install -e .    # editable install from src/
```

## Commands

```bash
# Tests
pytest

# CLI
stereo-depth preview   --path /dev/video0 --width 2560 --height 720 --fps 30
stereo-depth capture   --out-dir data/calib/$(date +%Y-%m-%d)_run1 --path /dev/video0
stereo-depth calibrate --data data/calib/charuco_2026-02-14_run1 --out outputs/calib/calib.yaml \
  --square-length 0.03 --marker-length 0.022 --dict-name DICT_5X5_100 --min-views 10
stereo-depth rectify   --calib outputs/calib/calib.yaml --data data/calib/charuco_2026-02-14_run1 \
  --out outputs/rectify_test --preview
stereo-depth depth     --calib outputs/calib/calib_strict.yaml \
  --left <left.png> --right <right.png> --out outputs/depth/demo --preset indoor \
  --matcher sgbm   # or: --matcher retinify
stereo-depth stream    --calib outputs/calib/calib_strict.yaml

# Mono (single-camera) intrinsics for Isaac Sim — see docs/mono_calibration.md
stereo-depth devices --probe
stereo-depth capture-mono   --out-dir data/mono/$(date +%Y-%m-%d)_run1 --path /dev/video0 \
  --target-views 40
stereo-depth calibrate-mono --data data/mono/2026-08-03_run1 --out outputs/calib/mono.yaml \
  --model auto   # or: pinhole | rational | fisheye
stereo-depth undistort      --calib outputs/calib/mono.yaml \
  --images data/mono/2026-08-03_run1 --out outputs/undistorted --alpha 0
```

## Architecture

See `docs/architecture.md` for full details. Layers (dependencies point inward):
`cli` → `app` → `adapters` → `use_cases` → `entities`

```
src/stereo_depth/
├── entities/        # pure data: FramePair, CalibrationResult, DepthMap
├── use_cases/       # abstract ports (ports.py) + StereoPipeline (pipeline.py)
├── adapters/        # concrete implementations: camera/, calibration/, rectifier/, matcher/, depth/
├── app/             # orchestrates adapters per command: calibrate.py, depth.py, rectify.py, stream.py
├── cli/             # Typer CLI commands → delegates to app/
├── infrastructure/  # cross-cutting utils: config/, io/, viz/
└── utils/           # logging, type aliases
```

**Rule:** `entities/` and `use_cases/` must never import from `adapters/`, `app/`, `cli/`, or `infrastructure/`.

**To swap matchers:** only change the concrete class in `app/depth.py` (`_build_matcher()`).

**Mono calibration** is a separate track from the stereo pipeline (different
entity, port, and YAML schema — it does not reuse `CalibrationResult`):
`adapters/calibration/mono_calibrator.py` (model fits + fair selection),
`capture_gates.py` (auto-collection logic), `isaac_export.py` (USD params),
`mono_yaml_repo.py`; orchestrated by `app/calibrate_mono.py` and
`app/undistort_mono.py`.

**Key invariant:** `K`/`D` pair with RAW images; `K_new` (zero distortion)
pairs with UNDISTORTED images. Never mix — see `docs/mono_calibration.md`.

## Testing

All tests are hardware-independent (FileSource or synthetic data).

| Test file | Covers |
|-----------|--------|
| `test_charuco_detect_synth.py` | Board detection on synthetic ChArUco image |
| `test_rectify.py` | Epipolar alignment < 2 px after rectification |
| `test_depth.py` | SGBM disparity sign + median accuracy on synthetic pair |
| `test_id_matching.py` | `_match_ids_one_view` common-ID extraction |
| `test_camera_stream.py` | `FileSource.stream()` count, type, shape |
| `test_pipeline_integration.py` | Full `StereoPipeline.process()` (skipped if files absent) |
| `test_retinify.py` | retinify adapter + matcher (skipped if `retinify` not installed) |
| `test_cli_preview.py` | CLI smoke test (no camera) |
| `test_mono_calib_synth.py` | mono model fits + fair model selection on synthetic views |
| `test_undistort_mono.py` | `K_new` pairing invariant across all three models |
| `test_isaac_export.py` | USD/OpenCV parameter conversion round-trips |
| `test_capture_gates.py` | coverage, blur, steadiness, tilt, auto-collect policy |
| `test_v4l2_devices.py` | `v4l2-ctl --list-formats-ext` parsing + device listing |
| `test_cli_mono.py` | `calibrate-mono` / `undistort` end-to-end on rendered boards |
| `test_capture_mono_live.py` | live capture loop via fake camera + fake clock |
