# Stereo Calibration

## Board

ChArUco board: `DICT_5X5_100`, 0.03 m square, 0.022 m marker.

Target reprojection error: **RPE < 0.5 px**.

---

## Capture Tips

- Collect 20–30 pairs minimum (30–60 recommended)
- Vary angles and distances (30–120 cm)
- Cover different image regions (corners, edges, centre)
- Keep pattern fully in frame and in focus

Avoid:
- Blurry images
- Only front-facing views
- Uniform distance/angle

---

## Commands

Collect images:
```bash
stereo-depth collect --path /dev/video0
```
Controls: `SPACE` to save a stereo pair, `q` to quit.

Run calibration:
```bash
stereo-depth calibrate \
  --data data/calib/<session> \
  --out outputs/calib/calib.yaml \
  --square-length 0.03 \
  --marker-length 0.022 \
  --dict-name DICT_5X5_100 \
  --min-views 10
```

---

## Output

Images saved to:
```
data/calib/<session>/left/
data/calib/<session>/right/
```

Calibration output:
```
outputs/calib/calib.yaml
outputs/calib/calib.report.json
```

---

## YAML Schema

```yaml
image_size: {width: W, height: H}
K1: [...]        # 9 floats, row-major (left intrinsics)
D1: [...]        # 5 floats (left distortion)
K2: [...]        # right intrinsics
D2: [...]        # right distortion
R:  [...]        # 9 floats, rotation right w.r.t. left
T:  [...]        # 3 floats, translation in metres
baseline_m: 0.060
R1: [...], R2: [...]   # rectification rotations
P1: [...], P2: [...]   # projection matrices (12 floats each)
Q:  [...]              # 16 floats, disparity-to-depth matrix
metrics:
  mono_reproj_L: 0.3
  mono_reproj_R: 0.3
  stereo_rms: 0.4
  used_views: 25
  matched_views: 25
```

Any schema change: bump `CalibrationResult` dataclass + `yaml_repo.py` + `MINOR` version.
