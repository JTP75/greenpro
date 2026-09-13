# Current status

Last updated 2026-09-13.

## Built

Capture → segment → geometry/downscale → local preview → **network delivery
to Will's simulator, verified end-to-end on real hardware** (see "Verified
this session" below). `greenpro/{config,capture,geometry,segmenters,
pipeline,server,patterns,sink}.py` and `run.py` are all written; the full
chain — camera → 17×9 grid → live `hazel-toad` instance — was run together
on the Pi (`pacel-rbp01`) on 2026-09-13.

- `config.py` — dataclass config tree, YAML loading, live-patchable via
  `LiveConfig.apply_patch`. The 30 FPS display ceiling
  (`MAX_DISPLAY_FPS`) is enforced here on every load/patch, now for three
  independent fields: `capture.fps`, and (new) `sink.fps`.
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
- `patterns.py` *(new)* — synthetic 17×9 grid generators (`blink`, `corners`,
  `bar`, `rainbow`, `solid`) plus `PatternPipeline`, a duck-typed sibling of
  `Pipeline` that publishes them straight into the same `LatestFrameHolder`,
  bypassing geometry entirely so a hand-crafted grid maps pixel-for-pixel
  onto display cells. Selected via `--source pattern:<name>`.
- `sink.py` *(new)* — `WebDisplaySink` (stdlib `urllib.request`, POSTs to
  `/api/i/<slug>/frame` per [simulator.md](simulator.md)) and `FrameSender`,
  a background thread mirroring `Pipeline`'s start/stop shape that sends
  each new grid, live-reconfigurable via `POST /config`, with failure
  backoff that never kills the thread or the source pipeline.
- `server.py` — `/`, `/raw.mjpg`, `/mask.mjpg`, `/grid.mjpg`, `/frame.json`,
  `/sink.json` *(new)*, `/config` (GET+POST).

## Verified this session (network path, synthetic content)

Ran `python run.py --source pattern:blink --send --instance hazel-toad` from
this dev machine (no camera, no Pi — see docs/simulator.md for why patterns
bypass the segmenter/geometry chain) and watched
`https://sundai.willsarg.com/hazel-toad?view=river` live. Results:

- **The network path works end-to-end.** Thousands of frames sent over an
  extended run; user confirmed visually seeing a red pixel top-left and a
  blue pixel bottom-right blinking at ~1 Hz, matching `patterns.blink()`
  exactly.
- **Orientation confirmed, not just assumed.** Because the display is a
  non-square 17×9 rectangle, a 90°/270° rotation mismatch is geometrically
  impossible; the only possible bugs are axis flips, and a lit diagonal pair
  rules out all four flip combinations at once. The observed red-top-left /
  blue-bottom-right result confirms `docs/protocol.md`'s "row 0 = top,
  col 0 = left" convention holds true all the way through the simulator's
  render. (This is the simulator's own mapping, not the physical camera's
  mount rotation on the Pi — that decision, below, is still open.)
- **Live reconfiguration works.** `POST /config {"sink":{"enabled":false}}`
  froze the send count immediately; re-enabling resumed it, no restart.
- **The 30 FPS ceiling holds for `sink.fps` too.** `POST /config
  {"sink":{"fps":120}}` came back clamped to `30` — third independent
  enforcement point, alongside `capture.fps` and the pipeline's own pacer.
- **Failure handling survives real failures**, both injected and organic:
  pointing `sink.instance` at a nonexistent slug produced repeated `400
  {"error":"bad instance name"}` with visibly increasing backoff (measured
  send rate collapsed from ~8/s to ~0.1/s) while the source pipeline kept
  publishing frames at its own unaffected rate throughout; two transient
  real failures also occurred organically during the run (a TLS handshake
  timeout, and a Cloudflare "error code: 1101") and both self-recovered on
  the next attempt with no intervention. Restoring the correct instance
  recovered sending automatically.
- **Found and fixed a real bug in the process:** `FrameSender` snapshotted
  `sink` config once per loop iteration, including across the backoff sleep
  — so a live fix during an active (up to 10s) backoff didn't take effect
  until the *next* iteration, even though `/sink.json`'s `target` field had
  already updated a cycle earlier. Fixed by re-reading config after the
  backoff wait, immediately before constructing the request.
- **Found and worked around a protocol gotcha:** the simulator's Cloudflare
  front-end returns `403` ("error code: 1010") for `urllib.request`'s
  default `Python-urllib/x.y` User-Agent — any ordinary-looking UA clears it.
  `WebDisplaySink` now always sends one. Documented in
  [simulator.md](simulator.md); this would bite anyone hand-rolling a client
  the way the shipped `docs/simulator.md` `requests` snippet does not (it
  happens to send `requests`' own default UA, which is not blocked).
- Also newly confirmed: `/frame.json`'s shape (153 entries, row-major) is
  correct against a running server (former item 6 below) — `PatternPipeline`
  exercises the exact same `LatestFrame`/`/frame.json` path as `Pipeline`.

## Verified this session (integrated run, real camera + live simulator)

Ran `python run.py --send --instance hazel-toad` on the Pi
(`pacel-rbp01`) with the real camera — the two halves (video pipeline and
network sender) running together for the first time as one process.
Observed live via `/frame.json`, `/sink.json`, and the log:

- **The integrated path works end-to-end.** Camera → nanodet →
  PP-HumanSeg → geometry → EMA smoothing → 17×9 grid → POST
  `https://sundai.willsarg.com/api/i/hazel-toad/frame`. Thousands of
  frames sent over a sustained run; a person silhouette was visible in the
  grid (`/frame.json`) and the sender logged HTTP 204s continuously.
- **The three-thread split held as designed:** display at ~20fps while
  inference ran at ~6.5fps (matches the `models.md` composed-cost estimate),
  with the sender pacing at ~12fps.
- **Found and fixed a real sender bug:** the `greenpro-sender` thread
  *died* mid-run on an unhandled `TimeoutError` (a read timeout on
  `urllib` — raised directly, not wrapped in `URLError`, so
  `WebDisplaySink.send()`'s handlers missed it and the thread was killed
  with "Exception in thread greenpro-sender"). The send counter froze at
  1158 while the pipeline kept running. Fixed in `6fad051` by catching
  `TimeoutError`/`OSError` in `send()` and converting them to `SinkError`
  so `FrameSender`'s backoff path handles them — a transient network
  failure must never kill the sender. Verified after redeploy: counter
  advanced continuously, zero tracebacks.

## Not yet verified

The network half's open items live in [simulator.md](simulator.md)'s
"Open questions" (exact validation messages for other malformed bodies,
whether `/frame` has its own rate limit, instance expiry, and the
still-unexplained "error code: 1101" seen once in the earlier run).

1. Geometry defaults (`rotation: 90, fit: crop`) are still a starting
   guess, not a decision — needs a live sweep against `/raw.mjpg` once the
   camera is physically mounted, to see how it's actually oriented. (The
   integrated run's silhouette was correctly upright, so the current
   values are at least self-consistent.)
2. `MotionSegmenter` should work out of the box (pure OpenCV, no model).
   Not yet run against a live feed to tune `var_threshold` /
   `min_blob_area`.
3. `NeuralSegmenter` has **no model file yet** — `models/` is empty and
   gitignored. Need to source or convert a MediaPipe Selfie Segmentation
   ONNX export (or fall back to a person-detector, per the contingency
   noted in `segmenters.py`'s `NeuralSegmenter` docstring) before
   `segmenter: neural` or `combo` can be exercised. (Not a blocker: the
   default `topdown` backend uses the vendored nanodet + PP-HumanSeg
   weights, which are git-tracked via LFS and present on the Pi.)
4. Throughput at `combo` segmenter settings on the Pi 5 is unmeasured.

## Explicitly out of scope for this repo (so far)

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
