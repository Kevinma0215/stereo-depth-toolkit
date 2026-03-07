# Hardware Setup

Camera: HBVCAM-W202011HD (USB 2.0 UVC, no vendor SDK required).

## Device Mapping

| Device | Description |
|--------|-------------|
| `/dev/video0` | SBS stitched stereo stream (2560x720) |
| `/dev/video2` | Left camera only |
| `/dev/video3` | Right camera only |

List available devices:
```bash
ls -l /dev/video*
```

Check supported formats:
```bash
sudo v4l2-ctl -d /dev/video0 --list-formats-ext
```

---

## Recommended Mode

Set MJPEG format for stable streaming at 30 fps:

```bash
sudo v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=2560,height=720,pixelformat=MJPG

sudo v4l2-ctl -d /dev/video0 --set-parm=30
```

---

## Notes

- Use `/dev/video0` for stereo processing (SBS split handled by `SBSSplitter`)
- If left/right appear swapped, add `--swap-lr` to any CLI command
- USB 2.0 bandwidth limits framerate at full 2560x720; MJPEG compression is required
