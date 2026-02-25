# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

A **production-quality Python library** for real-time stereo depth estimation using the HBVCAM-W202011HD camera.
- Primary input: stereo camera image pair (left + right)
- Primary output: 2D depth map (float32, metres); future: 3D point cloud
- Key design constraint: swapping the disparity matcher (e.g. SGBM → Retinify) must require **zero changes** outside `adapters/matcher/`

## Environment Setup

```bash
conda env create -f environment.yml
conda activate CV
pip install -e .
```

Conda env: `CV`, Python 3.10. Package installed editable from `src/`.

## Commands

```bash
# Tests
pytest
pytest tests/test_charuco_detect_synth.py::test_detect_charuco_on_synth

# CLI (thin wrapper over library — delegates to app/)
stereo-depth preview   --path /dev/video0 --width 2560 --height 720 --fps 30
stereo-depth collect   --path /dev/video0
stereo-depth calibrate --data data/calib/charuco_2026-02-14_run1 --out outputs/calib/calib.yaml \
  --square-length 0.03 --marker-length 0.022 --dict-name DICT_5X5_100 --min-views 10
stereo-depth rectify   --calib outputs/calib/calib.yaml --data data/calib/charuco_2026-02-14_run1 \
  --out outputs/rectify_test --preview
stereo-depth depth     --calib outputs/calib/calib_strict.yaml \
  --left <left.png> --right <right.png> --out outputs/depth/demo --preset indoor \
  --matcher sgbm   # or: --matcher retinify
```

## Hardware

- `/dev/video0` — SBS 2560 × 720 stream (USB 2.0 UVC, no vendor SDK)
- `/dev/video2` — left channel only
- `/dev/video3` — right channel only
- Use `--swap-lr` if left/right appear swapped

---

## Architecture — Clean Architecture

Four concentric layers; dependencies always point **inward**:
`cli` → `app` → `adapters` → `use_cases` → `entities`

```
src/stereo_depth/
│
├── entities/               # Layer 1 — pure data, ZERO imports from this project
│   ├── frame.py            # FramePair, RectifiedPair
│   ├── calibration.py      # CalibrationResult (14 fields)
│   └── depth.py            # DepthMap, PointCloud
│
├── use_cases/              # Layer 2 — abstract ports + pipeline
│   ├── ports.py            # ABCs: ICameraSource, ICalibrationRepo,
│   │                       #       IRectifier, IDisparityMatcher, IDepthEstimator
│   └── pipeline.py         # StereoPipeline: wires Rectifier → Matcher → DepthEstimator
│
├── adapters/               # Layer 3 — concrete implementations of ports.py
│   ├── calibration/
│   │   ├── charuco_calibrator.py  # ChArUco detection + stereoCalibrate + rectify maps
│   │   ├── yaml_repo.py           # ICalibrationRepo: load/save CalibrationResult ↔ YAML
│   │   └── retinify_adapter.py    # calibration_result_to_retinify() → dict
│   ├── camera/
│   │   ├── uvc_source.py          # open_source() + UVCSource (live camera)
│   │   └── file_source.py         # FileSource (offline / tests)
│   ├── rectifier/
│   │   └── opencv_rectifier.py    # IRectifier via cv2.remap (R1/R2/P1/P2)
│   ├── matcher/
│   │   ├── sgbm_matcher.py        # IDisparityMatcher — OpenCV SGBM (CPU fallback)
│   │   └── retinify_matcher.py    # IDisparityMatcher — Retinify TensorRT (primary)
│   └── depth/
│       └── opencv_depth_estimator.py  # IDepthEstimator via cv2.reprojectImageTo3D
│
├── app/                    # Application layer — orchestrates adapters for each command
│   ├── calibrate.py        # run_calibrate_charuco_stereo()
│   ├── depth.py            # run_depth_once(); builds StereoPipeline
│   └── rectify.py          # run_rectify_dataset()
│
├── cli/                    # Thin Typer commands — parse args → call app/
│   ├── app.py
│   ├── calibrate_cmd.py, capture_cmd.py, depth_cmd.py
│   ├── preview_cmd.py, rectify_cmd.py
│
├── infrastructure/         # Cross-cutting utilities (no business logic)
│   ├── config/             # io.py: save_yaml/load_yaml; schema.py: validation
│   ├── io/                 # pairs.py, sbs_capture.py (SBSSplitter), sinks.py (VideoRecorder)
│   └── viz/                # preview.py (preview_sbs), overlay.py
│
└── depth/                  # Legacy depth helpers still used by sgbm_matcher
    └── presets.py          # Named SGBM parameter presets (indoor / outdoor / …)
```

> **Rule:** `entities/` and `use_cases/` must never import from `adapters/`, `app/`, `cli/`, or `infrastructure/`.
> Verify: `grep -r "from stereo_depth.adapters" src/stereo_depth/entities/` must be empty.

---

## Key Data Structures

### `CalibrationResult` (`entities/calibration.py`)
```python
@dataclass
class CalibrationResult:
    image_size: tuple[int, int]  # (width, height)
    K1, D1: np.ndarray           # left intrinsics (3×3), distortion (5,)
    K2, D2: np.ndarray           # right intrinsics
    R: np.ndarray                # 3×3 rotation right w.r.t. left
    T: np.ndarray                # translation (3,) in metres
    baseline_m: float
    R1, R2: np.ndarray           # rectification rotations
    P1, P2: np.ndarray           # projection matrices (3×4)
    Q: np.ndarray                # 4×4 disparity-to-depth
    rpe_px: float                # reprojection error (informational)
```

### `DepthMap` (`entities/depth.py`)
```python
@dataclass
class DepthMap:
    data: np.ndarray              # float32 (H, W), metres, NaN = invalid
    disparity: np.ndarray         # float32 (H, W)
    left_rect: Optional[np.ndarray] = None   # uint8 BGR (H, W, 3)
    right_rect: Optional[np.ndarray] = None  # uint8 BGR (H, W, 3)
```

---

## Abstract Ports (`use_cases/ports.py`)

```python
class IRectifier(ABC):
    def rectify(self, pair: FramePair, calib: CalibrationResult) -> RectifiedPair: ...

class IDisparityMatcher(ABC):
    def compute(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        """Returns float32 disparity map, shape (H, W)."""

class IDepthEstimator(ABC):
    def to_depth(self, disparity: np.ndarray, calib: CalibrationResult) -> DepthMap: ...

class ICameraSource(ABC):
    def grab(self) -> FramePair: ...

class ICalibrationRepo(ABC):
    def load(self, path: str) -> CalibrationResult: ...
    def save(self, result: CalibrationResult, path: str) -> None: ...
```

**To swap matchers:** only change the concrete class injected in `app/depth.py` (`_build_matcher()`). Nothing else changes.

---

## Disparity Matchers

| Matcher | File | Backend | Notes |
|---------|------|---------|-------|
| `SgbmMatcher` | `adapters/matcher/sgbm_matcher.py` | OpenCV CPU | Fallback; presets from `depth/presets.py` |
| `RetinifyMatcher` | `adapters/matcher/retinify_matcher.py` | TensorRT GPU | Primary; raises `ImportError` at module level if `retinify` absent |

`RetinifyMatcher` maps preset names `'fast'/'balanced'/'accurate'` to `retinify.DepthMode`.
`retinify_adapter.py` converts `CalibrationResult` → retinify calibration dict.

CLI: `--matcher sgbm` (default) or `--matcher retinify`.

---

## Pipeline Data Flow

```
ICameraSource.grab()
  └─▶ FramePair
        └─▶ IRectifier.rectify()
              └─▶ RectifiedPair (left_rect, right_rect)
                    └─▶ IDisparityMatcher.compute()
                          └─▶ disparity float32 (H, W)
                                └─▶ IDepthEstimator.to_depth()
                                      └─▶ DepthMap (data, disparity, left_rect, right_rect)
```

---

## Calibration Workflow

Board: **ChArUco** (DICT_5X5_100, 0.03 m square, 0.022 m marker).
Target: RPE < 0.5 px. Collect 20–30 pairs at varied positions, distances 30–120 cm.

**YAML schema** (`outputs/calib/calib.yaml`):
```yaml
image_size: {width: W, height: H}
K1: [...]   # 9 floats, row-major
D1: [...]   # 5 floats
K2: [...], D2: [...]
R: [...], T: [...], baseline_m: 0.060
R1: [...], R2: [...], P1: [...], P2: [...], Q: [...]
metrics: {mono_reproj_L, mono_reproj_R, stereo_rms, used_views, matched_views}
```
Any schema change → bump `CalibrationResult` dataclass + `yaml_repo.py` + `MINOR` version.

---

## Data Layout

```
data/calib/<session>/left/    ← calibration PNG pairs
data/calib/<session>/right/
data/raw/sbs/                 ← raw SBS video files
outputs/calib/                ← calib.yaml + report.json
outputs/depth/<name>/         ← disparity.npy, depth_m.npy, disparity.png, left_rect.png, right_rect.png
```

---

## Testing

All tests are **hardware-independent** (use `FileSource` or synthetic data):

| Test file | What it covers |
|-----------|----------------|
| `test_charuco_detect_synth.py` | Board detection on a synthetically rendered ChArUco image |
| `test_rectify.py` | Epipolar alignment < 2 px after rectification |
| `test_depth.py` | SGBM disparity sign + median accuracy on synthetic shifted pair |
| `test_id_matching.py` | `_match_ids_one_view` common-ID extraction |
| `test_pipeline_integration.py` | Full `StereoPipeline` with real calib + image pair (skipped if files absent) |
| `test_retinify.py` | `retinify_adapter` unit tests; `RetinifyMatcher` tests skipped if `retinify` not installed |
| `test_cli_preview.py` | CLI smoke test (no camera) |
| `test_detect.py` | Additional detection tests |

`retinify` is optional — tests that require it are guarded with `@pytest.mark.skipif(find_spec("retinify") is None, ...)`.

---

## Versioning

Follows Semantic Versioning (MAJOR.MINOR.PATCH):
- `CalibrationResult` YAML schema change → bump `MINOR`
- Breaking public API change → bump `MAJOR`
- Tag releases: `git tag v0.2.0`; maintain `CHANGELOG.md`
