# Neural model selection and benchmarks

Referenced from `greenpro/detectors.py`, `greenpro/segmenters.py`, and
`greenpro/bench.py`. This is the evidence behind the current defaults in
`config.yaml` (`segmenter.kind: topdown`, `segmenter.detector_kind: nanodet`).

## Why not a single model

The obvious approach — one segmentation model, done — doesn't work at room
scale on a Pi 5 CPU:

- **YOLOv8n-seg / YOLO11n-seg** (instance segmentation): ~3000ms/frame on Pi 5
  CPU. The mask head is too expensive without an accelerator. Ruled out
  without running it here — the published benchmarks were unambiguous enough.
- **MediaPipe Selfie Segmentation**: ~116–128ms on ARM CPU (~8fps), and its
  Python wheels are unreliable on Python 3.13/aarch64. Also ruled out without
  running it here.
- **PP-HumanSeg alone**: fast, but trained on a *teleconferencing* dataset
  (291 videos, 23 conference scenes — see the
  [PP-HumanSeg paper](https://arxiv.org/abs/2112.07146)). It's a portrait/
  chest-up model; a small, distant, full-body figure is out of its training
  distribution.

The architecture that resolves this: run a **person detector** first (any
distance, any framing) to find *where* people are, then run PP-HumanSeg on
each detected person's **cropped box**. A tight crop resized to PP-HumanSeg's
192×192 input *is* portrait framing — exactly what it was trained on. This is
`TopDownSegmenter` in `segmenters.py`.

## Vendoring, not reimplementing

`vendor/opencv_zoo/` is a sparse checkout of
[opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) (Apache 2.0),
trimmed to three model directories, used **verbatim** rather than
reimplemented. `greenpro/vendor_loader.py`'s docstring has the full reasoning;
the short version: `person_detection_mediapipe/mp_persondet.py` embeds a
~2300-element literal SSD anchor array that must match the model's training
exactly, and there was no way to verify a hand-transcribed or regenerated
version of it against ground truth from inside this session. Using the
project's own tested file sidesteps that risk entirely. Each vendored file is
self-contained (only `numpy`/`cv2` imports) and loaded by file path via
`vendor_loader.load_module()`, not via `sys.path`.

## Benchmarked on the real Pi 5 (not extrapolated)

The original plan's numbers were all extrapolated from Raspberry Pi 4B
benchmarks (÷2.5 for the 4B→5 CPU difference). `greenpro/bench.py` replaced
those with real measurements, run via `.venv/bin/python -m greenpro.bench` on
`pacel-rbp01`, 15 iterations each, `cv2.dnn` backend:

| Model | Measured (Pi 5) | Notes |
|---|---|---|
| PP-HumanSeg, float32, single crop | **33.4ms** (29.9 fps) | int8bq fails to load, see below |
| NanoDet, float32, full frame @416² | **94.6ms** (10.6 fps) | int8bq fails to load |
| MediaPipe person detector, float32, full frame @224² | **49.8ms** (20.1 fps) | int8bq fails to load; box is unusable regardless, see below |

**Every `_int8bq` (block-quantized) variant in `opencv_zoo` fails to load**
under this Pi's OpenCV 4.10.0 build:

```
cv2.error: ... DequantizeLinear parse error ...
broadcast1D2TargetMat ... axis >= 0 && targetShape.size() > axis
```

This is a gap in that OpenCV build's ONNX importer for quantized ops, not a
problem with the model files. All defaults use the **float32** weights;
they're already fast enough that this isn't a real loss. If a future OpenCV
build fixes this, the quantized weights are still present in
`vendor/opencv_zoo/` and worth re-benchmarking.

## MediaPipe's person detector: box ≠ full body (found live, not in docs)

This cost real debugging time and is worth recording precisely.
`MediaPipePersonDetector`'s raw `(x1,y1,x2,y2)` output looked plausible on
paper — it's a real trained SSD person detector — but live testing on the
actual display showed it tracking as **a small square centered on the
face/chest**, cutting the rendered figure off below the shoulders.

The vendored `mp_persondet.py` itself hints at this — its own comment reads:

```python
# TODO: still don't know the meaning of face bbox
# each landmark: hip center point; full body point; shoulder center point; upper body point;
```

The model outputs a box *and* 4 auxiliary landmark points per detection.
MediaPipe's own pipeline (BlazePose-style) derives the actual full-body ROI
from those landmarks (center/size/rotation from specific landmark pairs) —
the raw box is only a coarse detection signal, not the final crop region.
`detectors.py` doesn't implement that landmark-to-ROI conversion (same
verification problem as the anchor table — no ground truth available in this
session to check a reconstruction against), so `MediaPipePersonDetector` is
**not the default** despite being ~2× faster. It's left in place, with this
limitation documented on the class itself, for anyone who wants to implement
that conversion properly.

**NanoDet** (`object_detection_nanodet`, a standard anchor-free COCO
detector filtered to the `person` class) has no such ambiguity — its boxes
are real object-extent boxes, head-to-feet for a standing person — and is
the default `detector_kind` for exactly this reason, the ~2× slower inference
notwithstanding.

## PP-HumanSeg under-segments legs (also found live)

Even given a **correct** full-body NanoDet box, PP-HumanSeg's own
webcam/teleconferencing training bias showed up immediately on the real
display: confident on torso/head/arms, but consistently under-confident on
legs, visually cutting the silhouette off around the waist/thighs.

Fix: `TopDownSegmenter` blends PP-HumanSeg's mask with a dim "floor" ellipse
spanning the *entire* detected box (`TopDownConfig.floor_fill`, default
`0.35`, via `max()` — see `segmenters.py`). Where PP-HumanSeg is confident,
its mask wins and shows full detail; where it isn't, the person doesn't
vanish, they render dimmer. Confirmed live: this took the rendered figure
from "cuts off at the torso" to a continuous head-to-floor silhouette.
`floor_fill: 0` disables this and uses PP-HumanSeg's mask as-is.

## Composed top-down cost

`bench.py`'s composed benchmark (detect + segment top-K boxes) was run
against a frame with no one clearly framed for detection, so its numbers are
confounded by variable box counts rather than pure compute cost — treat the
**derived** estimate below as more trustworthy than that specific run:

- MediaPipe detector (49.8ms) + K × PP-HumanSeg (33.4ms): K=1 → ~83ms
  (12fps), K=2 → ~116ms (8.6fps).
- NanoDet detector (94.6ms) + K × PP-HumanSeg (33.4ms): K=1 → ~128ms (7.8fps),
  K=2 → ~161ms (6.2fps) — matches live-measured `inference_fps` of 5-6.5fps
  with a person in frame.

This is well under the display's 20-30fps target, which is fine **only**
because of the pipeline's three-thread split (see `docs/video-pipeline.md`) —
inference runs on its own thread and the display free-runs off whatever mask
is latest, so a 6fps segmenter doesn't mean a 6fps display.

## Re-running the benchmark

```bash
ssh pacel@pacel-rbp01
cd ~/misc/greenpro
.venv/bin/python -m greenpro.bench --image some_frame.jpg --iters 20
# or, with no --image, captures one still from the live camera directly
# (only works if nothing else -- e.g. run.py -- currently holds the camera open)
```
