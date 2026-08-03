# Mono (single-camera) intrinsic calibration

Calibrate one wide-angle USB camera and export its intrinsics for
**NVIDIA Isaac Sim / Omniverse**.

Four commands:

```bash
stereo-depth devices                 # find the camera
stereo-depth capture-mono            # live, guided image collection
stereo-depth calibrate-mono          # fit + pick a distortion model + export
stereo-depth undistort               # optional distortion pre-processing
```

---

## 1. Find the camera

```bash
stereo-depth devices --probe
```

Lists every `/dev/video*` node with its sysfs name, USB id, supported
resolutions and frame rates. One physical camera usually registers several
nodes and only some deliver frames — `--probe` opens each and reads one frame
to mark which. Requires `v4l-utils` for the full format list
(`sudo apt install v4l-utils`); without it only the current mode is shown.

---

## 2. Collect images

```bash
stereo-depth capture-mono \
  --out-dir data/mono/$(date +%Y-%m-%d)_run1 \
  --path /dev/video0 \
  --width 1920 --height 1080 \
  --target-views 40
```

Images are saved **automatically** when a frame clears every quality gate —
no key pressing. The on-screen HUD tells you what to do next.

| Element | Meaning |
|---|---|
| Coverage grid (top-left) | green = covered, yellow outline = where the board is now, dim red = still missing |
| Status dots (below it) | `CORNERS` `SHARP` `STEADY` `NEW` `TILT` `EDGE` — the gates |
| Centre banner | the single most useful instruction right now |
| Bottom bar | `views / cells / tilts / edge` progress |

Keys: `SPACE` force-save (bypasses the novelty check), `R` undo the last save,
`G` toggle auto-capture, `Q` quit.

### The four gates and why they exist

- **Coverage** — the frame is split into a 4×4 grid (`--grid`); a cell counts
  as covered once ≥ 6 ChArUco corners land in it. Distortion coefficients are
  driven almost entirely by corners far from the image centre; a session shot
  only in the middle of the frame yields confident-looking coefficients that
  are pure extrapolation at the edges. This matters most on a wide lens.
- **Edge reach** — tracked separately from the grid, because grid occupancy
  on its own is misleading. The outer cells are wide, so a board parked in the
  middle of one marks it covered without the corners ever approaching the
  frame edge. A real session reported *all cells covered* while holding no
  data past 73% of the corner radius — exactly the region the coefficients
  depend on. The `EDGE` gate requires corners to reach `--edge-target`
  (default 0.85) of the way to the frame corners, and the session is not
  `done` until they do.
- **Sharpness** — Laplacian variance measured *only over the board's bounding
  box*, so a busy background cannot mask a blurred board.
- **Steadiness** — the board must barely move for several consecutive frames.
  Between-frame motion is a good proxy for during-exposure motion, and it
  catches smooth drift that a sharpness check alone passes.
- **Tilt diversity** — the board's pose is binned into
  `frontal / left / right / up / down`. All-frontal views are degenerate: fx,
  fy and the radial terms cannot be separated from one another. Tilts beyond
  ~50° are rejected instead, because heavy foreshortening localises corners
  badly.
- **Novelty** — a view is only kept if it reaches a new cell, fills a new tilt
  bin, or differs enough in position or apparent size from the last one. This
  stops the set filling with near-duplicates that bias the solver.

Aim for full grid coverage, `EDGE` green, and at least 4 tilt bins. Getting
`EDGE` green means deliberately letting part of the board hang off the side
of the frame — partial views still contribute their visible corners. The
command warns on exit about whichever of the three fell short.

---

## 3. Calibrate

```bash
stereo-depth calibrate-mono \
  --data data/mono/2026-08-03_run1 \
  --out outputs/calib/mono.yaml
```

By default (`--model auto`) it fits **all three** distortion models to the same
views and picks one:

| Model | Coefficients | Use |
|---|---|---|
| `pinhole` | `k1 k2 p1 p2 k3` | ordinary lenses, up to ~90° |
| `rational` | `k1..k6 p1 p2` | wide lenses, ~90–120° |
| `fisheye` | `k1..k4` (equidistant) | > 120° |

### How the winner is chosen

Adding coefficients *always* reduces training error, so training error is
never the deciding metric. Instead:

1. A held-out slice of views (25% by default) is excluded from every fit.
2. Any model is disqualified for implausible coefficients, a principal point
   far off centre, a horizontal FOV inconsistent with the others, or an
   undistortion map that folds back on itself inside the image.
3. Survivors are ranked by **holdout** reprojection error.
4. A more complex model only wins if it beats the simpler one by **>10%
   relative and >0.05 px absolute**. Ties go to the simplest model.
5. The winner is then refit on **all** views — the holdout existed only to
   decide.

The printed table and the `models:` block in the YAML show every model's
numbers and why anything was rejected, so the decision is auditable:

```
  model     coeffs  train RPE   holdout    HFOV  notes
 *pinhole        5     0.3421     0.371   68deg  ok
  rational       8     0.3418     0.370   69deg  ok
  fisheye        4     0.3418     0.369   69deg  undistortion not monotonic (fold-back)
```

Force a specific model with `--model fisheye` to skip selection entirely.

Target RPE is < 0.5 px. A `.report.json` is written next to the YAML — always,
including on failure — with collection statistics and per-image RPE so you can
find and re-shoot bad views.

---

## 4. Putting it into Isaac Sim

This is the part that most often goes wrong. **The images the sim renders and
the images your real pipeline consumes must have the same geometry.** There
are two consistent ways to arrange that, and mixing them is the bug.

### Route A — let the sim model the distortion (recommended)

Isaac Sim 4.x supports the OpenCV distortion models natively. Use the
`isaac_sim.opencv_fisheye` (or `opencv_pinhole`) block from the YAML, which
carries `fx, fy, cx, cy` in pixels plus the coefficients in OpenCV order.

- Sim renders a distorted image that matches the real camera.
- The real pipeline consumes **raw frames**, unmodified.
- Your downstream algorithm has to cope with distortion.

### Route B — undistort on both sides

Give the sim a plain pinhole camera built from **`K_new`** with zero
distortion (`isaac_sim.usd_camera_undistorted`), and run every real frame
through `stereo-depth undistort`.

- Both sides see straight lines.
- Costs you field of view (cropping) or leaves black borders, plus one remap
  per frame.
- Necessary if your downstream algorithm only understands pinhole cameras, or
  if you move to a simulator without fisheye support.

### Which one?

**It depends on what consumes the images, not on the lens.** If the downstream
algorithm handles distortion, Route A is more faithful and cheaper. If it
assumes a pinhole camera, Route B — and then the sim should see undistorted
images, because that is what the real system will be fed.

### The rule you cannot break

```
K, D      <->  RAW frames
K_new, 0  <->  UNDISTORTED frames
```

`K_new != K`. Undistortion changes the focal length and principal point.
Pairing raw `K` with undistorted images (or vice versa) produces a plausible
but wrong reconstruction. Every file this toolkit writes restates the pairing
in a `note:` field.

### USD camera conversion

`isaac_sim.usd_camera_*` gives the classic USD attributes, in millimetres:

```
focalLength              = fx * Ah / W
verticalAperture         = Ah * (H * fx) / (W * fy)
horizontalApertureOffset = (cx - W/2) * Ah / W
verticalApertureOffset   = (H/2 - cy) * Av / H
```

`Ah` is the assumed sensor width, defaulting to Isaac Sim's 20.955 mm. Only
the focal-to-aperture *ratio* affects projection, so this default is safe;
override it with `--sensor-width-mm` if you know the real sensor size.
`verticalAperture` is derived rather than assumed square, so a non-square pixel
aspect (fx ≠ fy) survives the conversion. The vertical offset carries a sign
flip because image *y* points down while USD film *y* points up.

---

## 5. Undistorting images

```bash
# batch
stereo-depth undistort --calib outputs/calib/mono.yaml \
  --images data/mono/2026-08-03_run1 --out outputs/undistorted --alpha 0

# live preview: raw | undistorted side by side
stereo-depth undistort --calib outputs/calib/mono.yaml --live --path /dev/video0
```

`--alpha` (pinhole/rational) and `--balance` (fisheye) trade field of view
against black borders:

- `0` — crop to fully valid pixels. Narrower FOV, no borders. **Default.**
- `1` — keep every source pixel. Full FOV, curved black borders.

In live mode, `a` and `z` step the value so you can see the trade-off; `s`
saves a snapshot.

Batch mode writes `undistorted_intrinsics.yaml` **into the output folder** —
that file holds `K_new` with zero distortion and is the one to use with those
images.

---

## Files written

| Path | Contents |
|---|---|
| `mono.yaml` | selected `model`, `K`, `D`, `rpe_px`; `models:` comparison; `undistort:` K_new at alpha 0 and 1; `isaac_sim:` USD + OpenCV blocks |
| `mono.report.json` | collection stats, selection reasoning, per-image RPE |
| `<out>/undistorted_intrinsics.yaml` | `K_new`, zero distortion, provenance |

The mono YAML carries `schema: stereo_depth/mono_intrinsics@1`. It is a
different format from the stereo `calib.yaml`; each loader rejects the other
with a clear message rather than a confusing `KeyError`.
