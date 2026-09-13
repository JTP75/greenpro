# Camera feed & video processing pipeline — session handoff

What was built, in what order, and what was learned by actually running it on
the Pi. This is a narrative handoff for this piece of the project (capture
through the 17×9 grid); it complements rather than replaces
[status.md](status.md) (living current-state tracker, owned by the
connectivity-side work happening in parallel) and
[models.md](models.md) (the benchmark data and model-selection reasoning this
doc points to rather than repeats).

## Scope

Everything from the camera to a `(17, 9, 3)` uint8 array: capture, human
segmentation, geometry/downscale, and a local preview server. Network
delivery to Will's Green Building simulator is a separate, now-finished piece
of work done in parallel — see [simulator.md](simulator.md) and
[protocol.md](protocol.md) for that side; this doc doesn't cover it.

## What exists, end to end

```
camera ─▶ capture thread ─▶ inference thread ─▶ composite thread ─▶ server
          (reads frames        (segments the        (downscale,        (MJPEG
           as fast as the       latest frame,         shade, EMA         previews,
           source allows;       drops stale ones,     smoothing,         /frame.json,
           applies rotation/    never queues)         paces to           live /config)
           flip here)                                 ≤30fps)
```

Three threads, deliberately decoupled (`greenpro/pipeline.py`):

- **capture** reads `(main_rgb, lores_gray)` from the camera (or a still/video
  file, for camera-free development) as fast as the source allows, and
  applies rotation/flip immediately — see "orientation-before-detection" below
  for why that has to happen *here*, not later.
- **inference** always processes the *latest* available frame — never a
  queue. A slow segmenter drops frames rather than falling behind. This is
  what lets a 6-20fps neural segmenter coexist with a 20-30fps display: they're
  different threads running at different, independent rates.
- **composite** is the display-facing loop: takes whatever raw frame and mask
  are currently latest, runs geometry (crop/squash/pad to the display's 9:17
  aspect), downscales to 17×9 via area-averaging, applies EMA smoothing over
  the grid (`SmoothingConfig.alpha`, default 0.6 — at 153 pixels a single
  flickering cell is very visible), shades it into RGB, and publishes. Paced
  to a hard ≥1/30s floor independent of config, per the display's rate limit.

Config (`greenpro/config.py`, `config.yaml`) is a nested dataclass tree,
loaded once and then live-patchable via `POST /config` — every field used to
be re-tunable without a restart, which mattered constantly while debugging
live (see below). `greenpro/server.py` exposes `/`, `/raw.mjpg`, `/mask.mjpg`,
`/grid.mjpg`, `/frame.json`, and `/config` (GET+POST).

## Segmentation: what's available, what's default

Four backends behind one `Segmenter.process(gray, rgb) -> float32 mask`
interface (`greenpro/segmenters.py`), selected by `segmenter.kind`:

- **`motion`** — MOG2 background subtraction. No model needed, but loses
  anyone who stops moving within seconds, and reacts to any motion (doors,
  lighting), not just people. This was the entire segmentation story at the
  start of this session.
- **`neural`** — a generic onnxruntime selfie-segmentation slot, superseded
  by `topdown` below; kept for anyone who wants to plug in a different model.
- **`boxfill`** — a person detector's boxes, each filled with a soft ellipse.
  Fast, no per-person segmentation cost.
- **`topdown`** (**current default**) — a person detector finds people at any
  distance, then PP-HumanSeg segments each detected crop. See
  [models.md](models.md) for the full reasoning; short version: PP-HumanSeg
  alone is a portrait/webcam model that fails at room scale, and this
  detect-then-segment pattern is the standard fix.
- **`combo`** — motion ∪ neural, kept for A/B comparison.

`greenpro/detectors.py` provides the person detectors `topdown`/`boxfill`
use: `nanodet` (default) and `mediapipe`. Both, and PP-HumanSeg, are vendored
verbatim from `opencv/opencv_zoo` into `vendor/opencv_zoo/` rather than
reimplemented — see that directory's role and the reasoning in
`greenpro/vendor_loader.py`'s docstring and [models.md](models.md).

## Three things found only by running it on real hardware

Each of these looked fine on paper and wasn't. Recording them here because
they're the kind of thing that's easy to reintroduce by accident later.

**1. Orientation has to be applied before detection, not just before display.**
The original design rotated the *output mask* to match the display's
orientation, downstream of segmentation. That's correct for what the display
shows, but wrong for what the detector sees: if the camera is physically
mounted sideways, the detector would be handed a person lying on their side
in the raw sensor frame — and detectors trained on upright photos degrade
badly on that. Fixed by moving `geometry.apply_orientation` into the capture
thread, applied once, immediately, before the frame reaches inference at all.
This turned "the camera has to be physically mounted the right way" into "set
`geometry.rotation` correctly" — relevant in the moment because the camera
ended up taped into position rather than properly mounted, with limited time
left in the session.

**2. MediaPipe's person-detector box is not a full-body box.** Confirmed live:
it tracked as a small square centered on the face/chest, cutting the render
off at the shoulders, no matter how the box was used downstream. Root cause
and why `nanodet` is the default instead: [models.md](models.md).

**3. PP-HumanSeg under-segments legs even given a correct box.** Also a
training-distribution problem (teleconferencing footage, chest-up framing) —
confirmed by watching the actual grid, not by reading the paper. Fixed with a
low-confidence "floor" ellipse blended under PP-HumanSeg's own mask over the
full detected box (`TopDownConfig.floor_fill`, default `0.35`) — full detail
where the model is confident, a dim guess rather than nothing where it isn't.
This was the last fix applied and moved the live result from "cuts off at the
torso" to a continuous head-to-floor silhouette.

## Running it

```bash
ssh pacel@pacel-rbp01
cd ~/misc/greenpro
.venv/bin/python run.py
# http://pacel-rbp01:8000/ from another machine on the same network
```

Camera-free, for iterating on geometry/shading without the Pi:
`python run.py --source image:some_frame.jpg`. See [setup.md](setup.md) for
the one-time environment setup (the `--system-site-packages` venv, and why
it's needed).

## State at end of session

Verified live, with a person actually in frame: the pipeline holds a
motionless subject (the original motivation for this work — motion detection
by definition loses anyone who stops moving), renders a continuous head-to-
floor silhouette, and the display stays at ~20fps while inference runs at
~5-6.5fps — the three-thread decoupling working as designed.

Not yet done: no multi-person test, no test at the far end of the intended
3-8m room-scale range (testing so far has been at close-to-medium range in a
small room), and `geometry.rotation`/`hflip`/`vflip` are set for the camera's
current taped-in-place position rather than a permanent mount. `models.md`'s
"composed cost" section notes its own benchmark run was against an
unrepresentative frame (no one clearly in view) — the derived estimate there
is more trustworthy than that specific run's numbers.
