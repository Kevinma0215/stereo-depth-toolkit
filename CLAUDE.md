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
