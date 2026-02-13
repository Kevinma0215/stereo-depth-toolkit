# Stereo Depth Toolkit

Toolkit for capturing and preparing stereo camera data for depth estimation.

## Current Features

✔ SBS stereo preview  
✔ calibration image collection  

---

## Hardware

This project uses a USB stereo camera exposing:

| Device | Purpose |
|--------|--------|
| /dev/video0 | SBS stitched stream |
| /dev/video2 | left camera |
| /dev/video3 | right camera |

---

## Environment Setup

```bash
conda create -n stereo-py310 python=3.10 -y
conda activate stereo-py310
cd stereo-depth-toolkit
python -m pip install -e .
````

---

## Preview Stereo Stream

```bash
stereo-depth preview --path /dev/video0 --width 2560 --height 720 --fps 30
```

If left/right appear swapped:

```bash
stereo-depth preview --path /dev/video0 --swap-lr
```

---

## Collect Calibration Images

Use a Charuco or checkerboard board.

```bash
stereo-depth collect --path /dev/video0
```

Controls:

* SPACE → save stereo pair
* q → quit

Output:

```
calib_data/
  left/
  right/
```

Capture 30–60 image pairs from different angles.

---

## Requirements

* Python 3.10
* OpenCV
* NumPy