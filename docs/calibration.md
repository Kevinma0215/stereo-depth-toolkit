# Stereo Calibration Image Collection

## Pattern

Use a Charuco or checkerboard pattern.

---

## Capture Tips

✔ vary angles and distance  
✔ cover different image regions  
✔ keep pattern in focus  

Avoid:

✗ blurry images  
✗ only front-facing views  

Collect 30–60 image pairs.

---

## Capture Command

```bash
stereo-depth collect --path /dev/video0
````

Output:

```
calib_data/
  left/
  right/
```