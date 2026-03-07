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
stereo-depth collect --path /dev/video0
```

Controls: `SPACE` to save a pair, `q` to quit. Collect 30–60 pairs from varied angles and distances (30–120 cm). Output saved to `calib_data/left/` and `calib_data/right/`.

### 3. Calibrate

Uses a ChArUco board (DICT_5X5_100, 0.03 m square, 0.022 m marker).

```bash
stereo-depth calibrate \
  --data data/calib/charuco_2026-02-14_run1 \
  --out outputs/calib/calib.yaml \
  --square-length 0.03 \
  --marker-length 0.022 \
  --dict-name DICT_5X5_100 \
  --min-views 10
```

Target: RPE < 0.5 px. See [docs/calibration.md](docs/calibration.md) for tips and YAML schema.

### 4. Rectify (optional verification)

```bash
stereo-depth rectify \
  --calib outputs/calib/calib.yaml \
  --data data/calib/charuco_2026-02-14_run1 \
  --out outputs/rectify_test \
  --preview
```

### 5. Single-frame depth estimation

```bash
stereo-depth depth \
  --calib outputs/calib/calib_strict.yaml \
  --left  data/calib/charuco_2026-02-14_run1/left/000000.png \
  --right data/calib/charuco_2026-02-14_run1/right/000000.png \
  --out outputs/depth/demo \
  --preset indoor \
  --matcher sgbm   # or: --matcher retinify
```

Outputs: `disparity.npy`, `depth_m.npy`, `disparity.png`, `left_rect.png`, `right_rect.png`

### 6. Live depth stream

```bash
stereo-depth stream --calib outputs/calib/calib_strict.yaml
```

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
