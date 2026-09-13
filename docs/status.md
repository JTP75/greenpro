# Current status

Last updated 2026-09-13.

## Built

Capture → segment → geometry/downscale → local preview → **network delivery
to Will's simulator, now verified end-to-end** (see "Verified this session"
below). `greenpro/{config,capture,geometry,segmenters,pipeline,server,
patterns,sink}.py` and `run.py` are all written; the capture/segment/geometry
half is still **not run against a live camera** (see "Not yet verified"), but
the network half has been run for real, against the live `hazel-toad`
instance, from this Windows dev machine with no camera and no Pi involved.

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

## Not yet verified

Everything below is about the capture/segment/geometry half specifically —
still untouched this session; the Pi is held by another session and was not
connected to. The network half's own open items moved to
[simulator.md](simulator.md)'s "Open questions" (exact validation messages
for other malformed bodies, whether `/frame` has its own rate limit, instance
expiry, and the still-unexplained "error code: 1101" seen once in this run).

1. Environment setup on the Pi (`apt install python3-opencv`, venv with
   `--system-site-packages`, `pip install onnxruntime pyyaml`) has not been
   run yet.
2. No end-to-end run against a real camera yet. `python run.py --source
   image:picam_snap.jpg` should be the first thing tried after syncing (the
   full server/pipeline/config wiring is now confirmed working via
   `pattern:` sources instead — see "Verified this session" above).
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
6. Throughput at `combo` segmenter settings on the Pi 5 is unmeasured.

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
