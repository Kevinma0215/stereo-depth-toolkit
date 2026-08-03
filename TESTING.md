# Testing & Verification Guide

How to verify this toolkit works — from a fresh environment, through the
automated suite, to a real camera and an intrinsic matrix in Isaac Sim.

Every command below has been run on this machine unless explicitly marked
**(needs camera)**.

---

## 0. Quick check

```bash
conda activate cv
python -c "import cv2, stereo_depth; print('OpenCV', cv2.__version__)"
pytest -q
stereo-depth --help
```

Expected:

```
OpenCV 4.13.0
256 passed, 14 skipped in ~22s
```

If any of the three fails, work through section 1.

---

## 1. Environment

### 1.1 Create

```bash
conda env create -f environment.yml
conda activate cv
pip install -e .
```

### 1.2 Verify

```bash
python -c "
import sys, cv2, numpy, yaml, typer, stereo_depth
print('Python  ', sys.version.split()[0])
print('OpenCV  ', cv2.__version__)
print('NumPy   ', numpy.__version__)
print('package ', stereo_depth.__file__)
"
which stereo-depth
```

Known-good on this machine:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| OpenCV | 4.13.0 |
| NumPy | 2.2.6 |
| typer | 0.27.1 |

### 1.3 OpenCV 5 will break this

`environment.yml` pins `opencv>=4.10,<5`. **Do not remove that pin** without
doing the port. With OpenCV 5.0.0 installed, 32 tests fail and 12 error:

- `cv2.fisheye.CALIB_*` constants moved to the `cv2` namespace, so the
  fisheye fit raises `AttributeError`.
- `cv2.stereoRectify` fails a `gemm` assertion, taking out `test_rectify.py`
  and `test_eval_calibration.py` — the pre-existing stereo path, not just
  the mono additions.

To check what you actually have, and to recover:

```bash
python -c "import cv2; print(cv2.__version__)"
conda install -n cv -c conda-forge 'opencv>=4.10,<5'   # if it says 5.x
```

---

## 2. Automated test suite

All tests are hardware-independent — no camera, no GPU, no calibration
files required. They use synthetic data or rendered boards.

### 2.1 Run everything

```bash
pytest -q
```

Expected: **256 passed, 14 skipped** (270 collected).

### 2.2 Useful subsets

```bash
pytest -v                             # per-test names
pytest -rs                            # explain every skip
pytest -x                             # stop at first failure
pytest tests/test_mono_calib_synth.py # one file
pytest -k "fisheye or selection"      # by name
pytest --durations=10                 # find slow tests
```

### 2.3 Coverage map

Mono intrinsic calibration (this feature):

| File | Tests | Covers |
|---|---:|---|
| `test_mono_calib_synth.py` | 19 | model fits, ground-truth recovery, fair model selection, sanity checks, holdout split |
| `test_undistort_mono.py` | 20 | the `K_new` pairing invariant across all three models |
| `test_capture_gates.py` | 33 | coverage grid, blur, steadiness, tilt bins, auto-collect policy |
| `test_cli_mono.py` | 25 | `calibrate-mono` / `undistort` end to end on rendered boards |
| `test_capture_mono_live.py` | 10 | live capture loop via fake camera + fake clock |
| `test_isaac_export.py` | 17 | USD / OpenCV parameter conversion round-trips |
| `test_v4l2_devices.py` | 18 | `v4l2-ctl` output parsing, device listing |

Pre-existing stereo pipeline:

| File | Tests | Covers |
|---|---:|---|
| `test_post_processor.py` | 20 | post-processing stage |
| `test_process_stack.py` | 19 | pipeline composition |
| `test_eval_calibration.py` | 18 | stereo calibration quality metrics |
| `test_hole_fill.py` | 13 | hole-filling post-processor |
| `test_retinify.py` | 12 | retinify adapter (5 skip without the package) |
| `test_depth_aggregator.py` | 10 | temporal depth aggregation |
| `test_rolling_buffer.py` | 7 | rolling frame buffer |
| `test_depth.py` | 6 | SGBM disparity sign + median accuracy |
| `test_cuda_bm_matcher.py` | 6 | CUDA block matcher (all 6 skip without a GPU) |
| `test_camera_streamer.py` | 5 | threaded camera streamer |
| `test_camera_stream.py` | 3 | `FileSource.stream()` |
| `test_rectify.py` | 3 | epipolar alignment < 2 px |
| `test_pipeline_integration.py` | 3 | full pipeline (3 skip without calibration data) |
| `test_charuco_detect_synth.py` | 1 | board detection on a synthetic image |
| `test_cli_preview.py` | 1 | CLI smoke test |
| `test_id_matching.py` | 1 | common-ID extraction |

`tests/test_detect.py` is an empty file and collects 0 tests.

### 2.4 The 14 skips are expected

```bash
pytest -q -rs
```

| Count | Reason | To enable |
|---:|---|---|
| 6 | `cv2.cuda.StereoBM_create` missing or no GPU | build OpenCV with CUDA |
| 5 | `retinify` not installed | install retinify (needs CUDA/TensorRT) |
| 3 | `data/calib/charuco_2026-02-14_run1/...` absent | supply that stereo dataset |

None of these block mono calibration.

---

## 3. End-to-end without a camera

Validates the whole calibration path before hardware arrives. It renders
board views through a **known** camera matrix, so you can check the numbers
that come back out.

### 3.1 Generate synthetic views

```bash
cd /path/to/stereo-depth-toolkit
mkdir -p /tmp/smoke
PYTHONPATH=tests python -c "
from test_cli_mono import _render_views
from pathlib import Path
print('rendered', _render_views(Path('/tmp/smoke/views')), 'views')
"
```

The ground truth these are rendered through is `fx = fy = 700.0`,
`cx = 472.0`, `cy = 366.0`, at 960x720, with no distortion.

### 3.2 Calibrate

```bash
stereo-depth calibrate-mono \
  --data /tmp/smoke/views \
  --out  /tmp/smoke/mono.yaml \
  --min-views 10
```

Actual output:

```
Distortion model comparison
  model     coeffs  train RPE   holdout    HFOV  notes
 *pinhole        5     0.3421     0.371   68deg  ok
  rational       8     0.3418     0.370   69deg  ok
  fisheye        4     0.3418     0.369   69deg  undistortion not monotonic (fold-back)

Selected: pinhole   (pinhole selected by holdout RPE: ...)
  image_size 960x720   views 24
  fx=700.65  fy=701.22  cx=473.90  cy=364.56
  RPE 0.3486 px
```

Three things to read here:

- **`fx=700.65` against a true 700.0** — the pipeline recovers the camera.
- **`*pinhole` won despite losing on raw error.** Rational and fisheye both
  scored *lower* holdout error (0.370, 0.369 vs 0.371), but neither cleared
  the >10% relative / >0.05 px absolute margin. This is the anti-overfitting
  rule doing its job: these images have no distortion, so nothing justifies
  extra coefficients.
- **fisheye was disqualified outright** for an undistortion that folds back
  inside the image — a nonsense fit, correctly rejected rather than ranked.

### 3.3 Undistort

```bash
stereo-depth undistort \
  --calib  /tmp/smoke/mono.yaml \
  --images /tmp/smoke/views \
  --out    /tmp/smoke/undistorted
```

Actual output:

```
Undistorted 24 image(s) -> /tmp/smoke/undistorted
  valid ROI at alpha=0.0: x=0 y=0 w=959 h=719
  fx=727.42 fy=728.94 cx=475.15 cy=364.52, distortion = 0
```

Note `fx` moved **700.65 → 727.42**. That is `K_new`, and it is the whole
point of section 6: these undistorted images pair with 727.42, while the raw
images pair with 700.65. Mixing them is the classic wide-angle bug.

### 3.4 Clean up

```bash
rm -rf /tmp/smoke
```

---

## 4. End-to-end with the real camera **(needs camera)**

### 4.0 Print the board

ChArUco, **7x5 squares, DICT_5X5_100**, square 0.03 m, marker 0.022 m.

Mount it on something rigid — foam board or clipboard. A bent or curled
board silently corrupts every view; the calibration cannot detect this and
will happily report a low RPE.

**Measure the printed squares with a ruler.** Printers routinely scale by
1–2%. Measure across several squares and divide. Pass the real value as
`--square-length`.

> This does not change the intrinsic matrix — `K` is in pixels and is
> scale-free — but it sets the metric scale of every pose, so it matters if
> you later use this calibration for distance measurement.

### 4.1 Find the camera

```bash
stereo-depth devices --probe
```

`--probe` opens each node and reads a frame. One physical camera usually
registers several `/dev/video*` nodes and only some deliver images — take
the one marked `[capture OK]` and note its highest resolution.

Without a camera attached you get:

```
No /dev/video* devices found.
Plug in the camera and re-run. On WSL or inside a container the USB device
may not be passed through.
```

If the format list is missing: `sudo apt install v4l-utils`.

### 4.2 Trial run first

Do a short run before committing to a full session — it catches a
mis-measured board, bad focus or bad lighting in two minutes instead of ten.

```bash
stereo-depth capture-mono \
  --out-dir data/mono/trial \
  --path /dev/videoN \
  --width 1920 --height 1080 \
  --target-views 15
```

Then calibrate it (section 4.4). If RPE is sane and detection was stable,
delete it and do the real run.

### 4.3 Full collection

```bash
stereo-depth capture-mono \
  --out-dir data/mono/$(date +%Y-%m-%d)_run1 \
  --path /dev/videoN \
  --width 1920 --height 1080 \
  --target-views 40 \
  --square-length 0.0298      # your measured value
```

Frames save **automatically** when every gate passes. Read the HUD:

| Element | Meaning |
|---|---|
| Grid, top-left | green = covered, yellow outline = board now, dim red = missing |
| Status dots | `CORNERS` `SHARP` `STEADY` `NEW` `TILT` |
| Centre banner | the single most useful instruction right now |
| Bottom bar | `views / cells / tilts` progress |

Keys: `SPACE` force-save · `R` undo · `G` toggle auto · `Q` quit.

**Aim for all 9 cells green and 4+ tilt bins.** Push the board into the
corners of the frame and tilt it 20–40° left, right, up and down. On a wide
lens the distortion coefficients come almost entirely from corners far from
the image centre; a session shot in the middle of the frame produces
confident-looking numbers that are pure extrapolation at the edges.

On exit it warns if coverage or tilt variety fell short.

### 4.4 Calibrate

```bash
stereo-depth calibrate-mono \
  --data data/mono/2026-08-03_run1 \
  --out  outputs/calib/mono.yaml \
  --square-length 0.0298
```

### 4.5 Pass criteria

| Check | Target | Where |
|---|---|---|
| RPE | **< 0.5 px** | `rpe_px`, and the summary line |
| Views used | ≥ 20 | `views_used` |
| Selected model | plausible for the lens | `model` |
| fx vs fy | within ~2% | `K` |
| cx, cy | near image centre | `K` |
| No model disqualified for a reason you don't expect | — | `models:` block |

Read the comparison table. If `fisheye` wins by a wide margin your lens
really is fisheye-projection; if `pinhole` wins, the extra coefficients were
not earning their place.

Per-image errors are in the report, worst first:

```bash
python -c "
import json
r = json.load(open('outputs/calib/mono.report.json'))
for v in sorted(r['per_view_rpe'], key=lambda x: -x['rpe_px'])[:10]:
    print(f\"{v['rpe_px']:7.3f} px  {v['view']}\")
"
```

Delete the worst offenders and re-run if a few images dominate the error.

### 4.6 Visual check

```bash
stereo-depth undistort --calib outputs/calib/mono.yaml --live --path /dev/videoN
```

Raw and undistorted side by side. Point it at a door frame or a table edge:
**straight lines in the world must be straight in the undistorted half**,
right out to the corners. `a` / `z` step alpha, `s` saves a snapshot.

If the edges still bow, coverage was insufficient — collect more views in
the corners and recalibrate.

---

## 5. Stereo pipeline (pre-existing)

Unchanged by the mono work, and still verifiable:

```bash
stereo-depth capture    --out-dir data/calib/$(date +%Y-%m-%d)_run1 --path /dev/video0 \
                        --width 2560 --height 720 --fps 30 --num-pairs 70
stereo-depth calibrate  --data data/calib/2026-08-03_run1 --out outputs/calib/calib.yaml \
                        --square-length 0.03 --marker-length 0.022 \
                        --dict-name DICT_5X5_100 --min-views 10 --per-image-rpe
stereo-depth rectify    --calib outputs/calib/calib.yaml --data data/calib/2026-08-03_run1 \
                        --out-dir outputs/rectify --preview
stereo-depth depth      --calib outputs/calib/calib.yaml \
                        --left L.png --right R.png --out outputs/depth/demo --preset indoor
stereo-depth stream     --calib outputs/calib/calib.yaml --fill-holes --fill-radius 5
```

Targets: stereo RPE < 0.5 px, epipolar error < 2 px after rectification.

---

## 6. The invariant to check before trusting anything

```
K, D      <->  RAW frames
K_new, 0  <->  UNDISTORTED frames
```

`K_new != K`. In section 3.3, undistortion moved `fx` from 700.65 to 727.42
on images with barely any distortion at all; on a real wide lens the gap is
far larger. Pairing raw `K` with undistorted images, or `K_new` with raw
frames, gives a reconstruction that looks plausible and is wrong.

Every file the toolkit writes restates this in a `note:` field. When you fill
in Isaac Sim, confirm which set you are holding:

| Route | Sim gets | Real pipeline feeds it |
|---|---|---|
| A — sim models the lens | `isaac_sim.opencv_fisheye` / `opencv_pinhole` | raw frames |
| B — undistort both sides | `isaac_sim.usd_camera_undistorted` (`K_new`, zero distortion) | `stereo-depth undistort` output |

Pick by what consumes the images, not by the lens. See
[docs/mono_calibration.md](docs/mono_calibration.md).

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 32 failed / 12 errors on a fresh env | OpenCV 5 installed | `conda install -n cv -c conda-forge 'opencv>=4.10,<5'` |
| `ModuleNotFoundError: stereo_depth` | package not installed | `pip install -e .` |
| `No /dev/video* devices found` | camera absent, or not passed through on WSL/container | attach the camera; check `lsusb` |
| `devices` shows no resolutions | `v4l2-ctl` missing | `sudo apt install v4l-utils` |
| Nothing auto-saves | a gate never goes green | read the banner; usually blur or novelty |
| `Move closer - too few corners` | board too small in frame | move in, or raise capture resolution |
| RPE > 1 px | blurred views, bent board, or wrong `--square-length` | check per-image RPE, drop the worst, re-shoot |
| `fisheye` chosen on a normal lens | too few edge views | fill the coverage grid and recalibrate |
| Undistorted corners still bow | insufficient corner coverage | collect views at the frame edges |
| `not a mono calibration file` | passed a stereo `calib.yaml` | use the file from `calibrate-mono` |
| Wrong scale in metric results | printer scaling | measure the squares, pass `--square-length` |

---

## 8. Repo hygiene

`__pycache__/`, `*.egg-info/`, `build/`, `dist/` are ignored. `pip install -e .`
should leave the working tree clean:

```bash
pip install -e . -q && git status --short   # expect no output
```
