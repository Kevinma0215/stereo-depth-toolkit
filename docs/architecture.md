# Architecture

Clean Architecture with four concentric layers. Dependencies always point **inward**:

`cli` → `app` → `adapters` → `use_cases` → `entities`

## Module Map

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
│   └── pipeline.py         # StereoPipeline: process() + stream()
│
├── adapters/               # Layer 3 — concrete implementations of ports.py
│   ├── calibration/
│   │   ├── charuco_calibrator.py  # ChArUco detection + stereoCalibrate + rectify maps
│   │   ├── yaml_repo.py           # ICalibrationRepo: load/save CalibrationResult ↔ YAML
│   │   └── retinify_adapter.py    # calibration_result_to_retinify() → dict
│   ├── camera/
│   │   ├── uvc_source.py          # open_source() + UVCSource + UvcSource
│   │   └── file_source.py         # FileSource: single-file or directory mode
│   ├── rectifier/
│   │   └── opencv_rectifier.py    # IRectifier via cv2.remap (R1/R2/P1/P2)
│   ├── matcher/
│   │   ├── sgbm_matcher.py        # IDisparityMatcher — OpenCV SGBM (CPU)
│   │   ├── sgbm_presets.py        # SGBMPreset dataclass + preset() factory
│   │   └── retinify_matcher.py    # IDisparityMatcher — Retinify TensorRT (GPU)
│   └── depth/
│       └── opencv_depth_estimator.py  # IDepthEstimator via cv2.reprojectImageTo3D
│
├── app/                    # Layer 4 — orchestrates adapters per command
│   ├── calibrate.py        # run_calibrate_charuco_stereo()
│   ├── depth.py            # run_depth_once(); builds StereoPipeline
│   ├── rectify.py          # run_rectify_dataset()
│   └── stream.py           # run_stream()
│
├── cli/                    # Thin Typer commands — parse args → call app/
│   ├── app.py
│   ├── calibrate_cmd.py, capture_cmd.py, depth_cmd.py
│   ├── preview_cmd.py, rectify_cmd.py, stream_cmd.py
│
├── infrastructure/         # Cross-cutting utilities (no business logic)
│   ├── config/             # io.py: save_yaml/load_yaml; schema.py: validation
│   ├── io/                 # pairs.py, sbs_capture.py (SBSSplitter), sinks.py (VideoRecorder)
│   └── viz/                # preview.py (preview_sbs), overlay.py
│
└── utils/                  # Shared helpers
    ├── logging.py
    └── types.py
```

**Rule:** `entities/` and `use_cases/` must never import from `adapters/`, `app/`, `cli/`, or `infrastructure/`.

```bash
# Verify:
grep -r "from stereo_depth.adapters" src/stereo_depth/entities/   # must be empty
grep -r "from stereo_depth.adapters" src/stereo_depth/use_cases/  # must be empty
```

---

## Key Data Structures

### `CalibrationResult` (`entities/calibration.py`)

```python
@dataclass
class CalibrationResult:
    image_size: tuple[int, int]  # (width, height)
    K1, D1: np.ndarray           # left intrinsics (3x3), distortion (5,)
    K2, D2: np.ndarray           # right intrinsics
    R: np.ndarray                # 3x3 rotation right w.r.t. left
    T: np.ndarray                # translation (3,) in metres
    baseline_m: float
    R1, R2: np.ndarray           # rectification rotations
    P1, P2: np.ndarray           # projection matrices (3x4)
    Q: np.ndarray                # 4x4 disparity-to-depth
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
    def stream(self) -> Iterator[FramePair]: ...

class ICalibrationRepo(ABC):
    def load(self, path: str) -> CalibrationResult: ...
    def save(self, result: CalibrationResult, path: str) -> None: ...
```

---

## Pipeline Data Flow

### Single frame (`StereoPipeline.process(pair)`)

```
ICameraSource.grab()
  └─> FramePair
        └─> IRectifier.rectify()
              └─> RectifiedPair (left_rect, right_rect)
                    └─> IDisparityMatcher.compute()
                          └─> disparity float32 (H, W)
                                └─> IDepthEstimator.to_depth()
                                      └─> DepthMap (data, disparity, left_rect, right_rect)
```

### Continuous stream (`StereoPipeline.stream()`)

```
ICameraSource.stream()
  └─> FramePair  --+
                   | (repeated for every frame)
                   v
             StereoPipeline.process()
                   └─> DepthMap  -> yield to caller
```

---

## Versioning

Semantic Versioning (MAJOR.MINOR.PATCH):
- `CalibrationResult` YAML schema change → bump `MINOR` + update dataclass + `yaml_repo.py`
- Breaking public API change → bump `MAJOR`
- Tag releases: `git tag v0.2.0`; maintain `CHANGELOG.md`
