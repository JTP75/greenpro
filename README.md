# greenpro

A live camera feed, reduced to a human silhouette, streamed to the MIT Green
Building's 17×9 RGB pixel display.

## What this is

The [Green Building](https://en.wikipedia.org/wiki/Green_Building_(MIT)) has a
17-row × 9-column RGB array (see [17x9-Tetris](https://github.com/) for one
existing project targeting it — a Tetris implementation whose `Display` /
`Frame` interface this project reuses for the eventual hand-off). This project
points a Raspberry Pi camera at a room and turns whoever's in frame into a
silhouette on that array, live:

1. **Capture** — a Raspberry Pi with a camera module grabs frames.
2. **Segment** — a human-figure mask is computed each frame (motion-based,
   neural, or both combined), producing a two-color image: figure vs.
   background.
3. **Downscale** — the mask is reduced to 17×9 by area-averaging, so a cell
   straddling the edge of a silhouette gets a proportionate blend between the
   two colors rather than a hard, aliased edge.
4. **Send** — the resulting 17×9×3 frame goes out to Will's Green Building
   display simulator/server (see [docs/simulator.md](docs/simulator.md) for
   its wire protocol).

This repo covers steps 1–3 in full, plus a local preview server so you can
watch the raw feed, the two-color mask, and the final 17×9 grid side by side
while tuning. Step 4 (network delivery to the building's server) is a small
adapter on top of this pipeline's output and is tracked separately — see
[docs/status.md](docs/status.md) for exactly where that boundary sits today.

## Hardware

- Raspberry Pi 5 (4GB), reachable over SSH as `pacel@pacel-rbp01`
- Camera: `ov5647` (Camera Module v1), connected via CSI
- Target display: MIT Green Building, 17 rows × 9 columns of RGB pixels,
  driven through a display server built by Will (Green Building simulator) —
  not part of this repo

## Repo layout

```
greenpro/
  run.py            entry point: python run.py [--source ...] [--config ...]
  config.yaml        every tunable parameter, documented inline
  models/            .onnx model weights (gitignored; see docs/setup.md)
  greenpro/
    config.py        Config dataclasses, YAML loading, live-patchable config
    capture.py        camera + image/video file frame sources
    geometry.py        rotate/flip/fit/crop + the 17x9 area-average downscale
    segmenters.py      motion, neural, and combined human-figure segmenters
    pipeline.py        background worker thread tying the above together
    server.py          local HTTP preview (MJPEG streams + /frame.json + live config)
  docs/               current project state, setup notes, the frame protocol,
                       and what's known about Will's simulator/server
```

## Quick start

Without a camera, using a still or a clip to iterate on geometry/shading:

```bash
pip install -r requirements.txt
python run.py --source image:picam_snap.jpg
# open http://localhost:8000/
```

On the Pi, with the real camera (see [docs/setup.md](docs/setup.md) for the
one-time environment setup — it needs a venv with system site packages so it
can see the apt-installed `picamera2` and `opencv`):

```bash
ssh pacel@pacel-rbp01
cd ~/misc/greenpro
.venv/bin/python run.py
# from another machine: http://pacel-rbp01:8000/
```

Every parameter in `config.yaml` — rotation, crop, colors, shading curve,
which segmenter is active, MOG2 thresholds — can also be changed live without
restarting:

```bash
curl -X POST http://pacel-rbp01:8000/config -d '{"geometry": {"rotation": 180}}'
```

## The 17×9 output contract

`GET /frame.json` returns:

```json
{"w": 9, "h": 17, "rgb": [[r,g,b], ...], "frame_no": 1234, "fps": 19.8}
```

`rgb` has exactly 153 entries, row-major (row 0 — the top of the display —
first). This is the hand-off point for the network delivery code. See
[docs/protocol.md](docs/protocol.md) for the full contract and a worked
example adapting it to the `Display`/`Frame` classes from `17x9-Tetris`.

## Display rate limit

The Green Building display must not be sent frames faster than **30 FPS**
(confirmed with the display's maintainer, and documented in the `17x9-Tetris`
README: `Display.send()` should be called at most once every 0.033s). This
pipeline enforces that in two independent places — config can't hold a value
above 30, and the capture loop paces itself to a ≥1/30s floor regardless of
what config claims — so nothing downstream needs to re-check it.

## Status

See [docs/status.md](docs/status.md) for what's built, what's verified on
real hardware, and what's still open (model selection, physical camera
mounting/orientation, the network-delivery adapter).

## License

MIT — see [LICENSE](LICENSE).
