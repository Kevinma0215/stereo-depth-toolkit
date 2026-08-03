# Stereo Depth Toolkit

Production-quality Python library for real-time stereo depth estimation using the HBVCAM-W202011HD USB stereo camera.

- Input: stereo image pair (left + right)
- Output: float32 depth map in metres
- Supports swappable disparity matchers (SGBM, Retinify) with zero pipeline changes

---

## Hardware

| Device | Purpose |
|--------|---------|
| `/dev/video0` | SBS stitched stream (2560x720) |
| `/dev/video2` | Left camera only |
| `/dev/video3` | Right camera only |

Use `--swap-lr` if left/right appear swapped. See [docs/hardware.md](docs/hardware.md) for device setup.

---

## Setup

```bash
conda env create -f environment.yml
conda activate cv
pip install -e .
```

Requires Python 3.10. The package is installed editable from `src/`.

---

## Workflow

### 1. Preview the stereo stream

```bash
stereo-depth preview --path /dev/video0 --width 2560 --height 720 --fps 30
```

Replay a recorded video instead:
```bash
stereo-depth preview --video sbs_scene_10s_mjpg.avi
```

### 2. Collect calibration images

```bash
# stereo-depth collect --path /dev/video0
stereo-depth capture \
    --out-dir data/calib/$(date +%Y-%m-%d)_run2\
    --path /dev/video0 \
    --width 2560 --height 720 --fps 30 \
    --num-pairs 70
```

Controls: `SPACE` to save a pair, `q` to quit. Collect 30–60 pairs from varied angles and distances (30–120 cm). Output saved to `calib_data/left/` and `calib_data/right/`.

### 3. Calibrate

Uses a ChArUco board (DICT_5X5_100, 0.03 m square, 0.022 m marker).

```bash
stereo-depth calibrate \
  --data data/calib/2026-03-08_run2 \
  --out outputs/calib/0308_try5/calib.yaml \
  --square-length 0.03 \
  --marker-length 0.022 \
  --dict-name DICT_5X5_100 \
  --min-views 10
  --per-image-rpe # optional for image selection
```

Target: RPE < 0.5 px. See [docs/calibration.md](docs/calibration.md) for tips and YAML schema.

### 4. Rectify (optional verification)

```bash
stereo-depth rectify \
  --calib outputs/calib/0308_try4/calib.yaml \
  --data data/calib/2026-03-08_run2 \
  --out-dir outputs/rectify/0308_try4 \
  --preview
```

### 5. Single-frame depth estimation

```bash
stereo-depth depth \
  --calib outputs/calib/0308_try4/calib.yaml \
  --left  data/calib/2026-03-08_run2/left/0035.png \
  --right data/calib/2026-03-08_run2/right/0035.png \
  --out outputs/depth/0308_try2 \
  --preset indoor \
  --matcher sgbm
```

Outputs: `disparity.npy`, `depth_m.npy`, `disparity.png`, `left_rect.png`, `right_rect.png`

### 6. Live depth stream

```bash
stereo-depth stream --calib outputs/calib/calib.yaml --fill-holes --fill-radius 5
```

### 7. Aggregator test
```bash
python tools/tune_sgbm.py --left data/calib_frames/left/left_00020.png --right data/calib_frames/right/right_00020.png --calib outputs/calib/calib_strict.yaml
```

### 8. Evaluate Depth Accuracy
```bash
# 1. capture at each distance
python tools/capture_accuracy_seq.py --distance 0.20 --frames 30
python tools/capture_accuracy_seq.py --distance 0.30 --frames 30
python tools/capture_accuracy_seq.py --distance 0.40 --frames 30
python tools/capture_accuracy_seq.py --distance 0.50 --frames 30
python tools/capture_accuracy_seq.py --distance 0.60 --frames 30

# 2. evaluate all at once
python tools/evaluate_depth.py --all
```

---

## Mono (single-camera) intrinsics

Separate track for calibrating **one** wide-angle USB camera, e.g. to drop into
NVIDIA Isaac Sim. Full guide: [docs/mono_calibration.md](docs/mono_calibration.md).

```bash
stereo-depth devices --probe                       # find the camera

stereo-depth capture-mono \                        # live, auto-collects good views
  --out-dir data/mono/$(date +%Y-%m-%d)_run1 --path /dev/video0 --target-views 40

stereo-depth calibrate-mono \                      # fits 3 models, picks one, exports
  --data data/mono/2026-08-03_run1 --out outputs/calib/mono.yaml

stereo-depth undistort \                           # optional pre-processing
  --calib outputs/calib/mono.yaml --images data/mono/2026-08-03_run1 \
  --out outputs/undistorted --alpha 0
```

`capture-mono` guides you on screen — it tracks image coverage, sharpness,
steadiness and board tilt, and saves frames automatically when all gates pass.

`calibrate-mono` fits `pinhole` (5 coeff), `rational` (8) and `fisheye` (4) to
the same views and selects by **held-out** reprojection error, so extra
coefficients cannot win by overfitting. The YAML carries both Isaac Sim USD
camera attributes and the OpenCV distortion blocks.

> **`K_new != K`.** `K`/`D` describe raw frames; `K_new` (zero distortion)
> describes undistorted frames. Mixing them is the classic wide-angle bug.

---

## Matchers

| Matcher | Flag | Backend | Notes |
|---------|------|---------|-------|
| `SgbmMatcher` | `--matcher sgbm` | OpenCV CPU | Default fallback |
| `RetinifyMatcher` | `--matcher retinify` | TensorRT GPU | Optional; install `retinify` separately |

Presets: `indoor`, `outdoor`, `high_quality` (SGBM); `fast`, `balanced`, `accurate` (Retinify).

---

## Architecture

Clean Architecture with four layers (dependencies point inward):
`cli` → `app` → `adapters` → `use_cases` → `entities`

See [docs/architecture.md](docs/architecture.md) for the full module map, data structures, and pipeline data flow.

---

## Data Layout

```
data/calib/<session>/left/     # calibration PNG pairs
data/calib/<session>/right/
data/raw/sbs/                  # raw SBS video files
outputs/calib/                 # calib.yaml + report.json
outputs/depth/<name>/          # disparity.npy, depth_m.npy, *.png
```

---

## Testing

All tests are hardware-independent. Run with:

```bash
pytest
```

Tests use `FileSource` or synthetic data. Tests requiring the `retinify` package or real calibration files are auto-skipped when absent.

---

## Recording & Converting Video

Record SBS stream:
```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 2560x720 -framerate 30 \
  -i /dev/video0 -t 10 -c:v libx264 -pix_fmt yuv420p sbs_scene_10s.mp4
```

Convert MP4 to AVI (MJPEG, for replay compatibility):
```bash
ffmpeg -y -i sbs_scene_10s.mp4 -c:v mjpeg -q:v 3 -an sbs_scene_10s_mjpg.avi
```
Add post-processing stage to the stereo pipeline following @docs/architecture.md
Use @docs/post_processor_reference.py as the implementation reference.