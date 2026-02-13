# Camera Device Setup

List video devices:

```bash
ls -l /dev/video*
````

Check formats:

```bash
sudo v4l2-ctl -d /dev/video0 --list-formats-ext
```

## Expected mapping

| Device      | Description       |
| ----------- | ----------------- |
| /dev/video0 | SBS stereo stream |
| /dev/video2 | left camera       |
| /dev/video3 | right camera      |

Use `/dev/video0` for stereo processing.

---

## Recommended mode

Set MJPG for stable streaming:

```bash
sudo v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=2560,height=720,pixelformat=MJPG

sudo v4l2-ctl -d /dev/video0 --set-parm=30
```