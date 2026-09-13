# Current status

Last updated 2026-09-13.

## Built

Capture → segment → geometry/downscale → local preview, per the plan in this
session. All of `greenpro/{config,capture,geometry,segmenters,pipeline,server}.py`
and `run.py` are written and believed correct, but **not yet run end-to-end**
(see "Not yet verified" below) — the code has not been synced to the Pi or
exercised against a live camera in this session.

- `config.py` — dataclass config tree, YAML loading, live-patchable via
  `LiveConfig.apply_patch`. The 30 FPS display ceiling
  (`MAX_DISPLAY_FPS`) is enforced here on every load/patch.
- `capture.py` — `PiCameraSource` (dual-stream picamera2: full-res RGB +
  ISP-scaled lores YUV, Y-plane taken directly as grayscale), plus
  `ImageSource`/`VideoSource` for camera-free development.
- `geometry.py` — rotation/flip, three fit modes (crop/squash/pad), zoom,
  and the 17×9 area-average downscale via `cv2.resize(..., INTER_AREA)`.
- `segmenters.py` — `MotionSegmenter` (MOG2 + morphology + blob filtering),
  `NeuralSegmenter` (ONNX Runtime, model not yet chosen — see below),
  `ComboSegmenter` (union/intersection/mean/motion-gated).
- `pipeline.py` — background thread, live-config-aware, publishes a
  `LatestFrame` (raw/mask/grid + timings) that HTTP handlers block on.
  Independent frame-rate pacer as the second enforcement of the 30 FPS cap.
- `server.py` — `/`, `/raw.mjpg`, `/mask.mjpg`, `/grid.mjpg`, `/frame.json`,
  `/config` (GET+POST).

## Not yet verified

Everything in the plan's "Verification" section is still open:

1. Environment setup on the Pi (`apt install python3-opencv`, venv with
   `--system-site-packages`, `pip install onnxruntime pyyaml`) has not been
   run yet.
2. No end-to-end run yet, camera-free or live. `python run.py --source
   image:picam_snap.jpg` should be the first thing tried after syncing.
3. Geometry defaults (`rotation: 90, fit: crop`) are a starting guess, not a
   decision — needs a live sweep against `/raw.mjpg` once the camera is
   physically mounted, to see how it's actually oriented.
4. `MotionSegmenter` should work out of the box (pure OpenCV, no model). Not
   yet run against a live feed to tune `var_threshold` / `min_blob_area`.
5. `NeuralSegmenter` has **no model file yet** — `models/` is empty and
   gitignored. Need to source or convert a MediaPipe Selfie Segmentation
   ONNX export (or fall back to a person-detector, per the contingency noted
   in `segmenters.py`'s `NeuralSegmenter` docstring) before `segmenter: neural`
   or `combo` can be exercised.
6. `/frame.json` shape (153 entries, row-major) has not been checked against
   a running server.
7. Throughput at `combo` segmenter settings on the Pi 5 is unmeasured.
8. The 30 FPS ceiling's two enforcement points are implemented but untested
   against an actual `POST /config {"fps": 120}` — should be a five-minute
   check once the server is running.

## Explicitly out of scope for this repo (so far)

- **Network delivery to Will's Green Building simulator/server.** The hand-off
  point (`Pipeline.latest_grid()` / `GET /frame.json`) is built and documented
  in [protocol.md](protocol.md), but nothing sends frames anywhere yet. The
  simulator's own wire protocol (`POST /api/i/<slug>/frame`, JSON shape) is
  now reverse-engineered and documented in [simulator.md](simulator.md) —
  writing the adapter is now just wiring `latest_grid()` output into the
  `send_grid()` example there.
- **Physical camera mounting.** Whether the camera is actually rotated 90°
  on its mount, and which way, is a physical decision that determines the
  right `geometry.rotation`/`hflip`/`vflip` values — needs to happen with the
  Pi in its installed location.

## Open decisions

- **Shading vs. strict two-color.** The original ask was for a two-color
  image (figure/background) *and* a shaded average at the grid level, which
  pull in slightly different directions. Current default (`shading: linear`)
  interpolates smoothly between `fg_color`/`bg_color` at the edges of a
  silhouette; `shading: threshold` snaps every cell to one of the two colors
  if the interpolated look reads as unwanted "third colors" on the real
  hardware. Worth deciding once it's visible on the actual array (behavior
  differs meaningfully at 153 pixels vs. on a monitor preview).
- **Segmenter model choice.** See point 5 above — needs a model file picked
  and, likely, tried against real footage before committing to it in
  `config.yaml`'s default.
