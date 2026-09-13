"""Three threads, decoupled by design:

  capture thread   -- reads (main_rgb, lores_gray) as fast as the source
                       allows, publishes to a "latest raw frame" slot.
  inference thread -- always processes the *latest* raw frame (never a
                       queue -- a slow segmenter drops frames rather than
                       falling behind), publishes a coverage mask.
  composite thread -- the display-facing loop. Runs geometry, EMA smoothing,
                       downscale, and shading against whatever raw frame and
                       mask are currently latest, and publishes LatestFrame
                       at up to the display's rate.

Why split it this way: the neural segmenters (see segmenters.py,
detectors.py) run well under display frame rate -- measured ~6-20fps on the
Pi 5 depending on backend and person count (docs/models.md) -- but the
display should still get a fresh, smoothly-paced frame at up to 30fps. Before
this split, one combined loop meant the segmenter's frame rate *was* the
display's frame rate. Now the composite thread free-runs off whatever mask is
newest, so a 10fps segmenter still drives a 20-30fps display.

The 30 FPS ceiling on frames sent to the display (see config.MAX_DISPLAY_FPS)
is enforced *here*, independently of whatever fps value config claims: the
composite loop is paced to a minimum period, so even a bug that let an
out-of-range fps through config would not be able to drive the display
faster than the hardware allows.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import capture, geometry, segmenters
from .config import MAX_DISPLAY_FPS, Config, LiveConfig

log = logging.getLogger(__name__)

MIN_FRAME_PERIOD = 1.0 / MAX_DISPLAY_FPS  # hard floor, independent of config


@dataclass
class StageTimings:
    capture: float = 0.0
    segment: float = 0.0
    geometry: float = 0.0
    total: float = 0.0
    fps: float = 0.0          # composite/display rate
    inference_fps: float = 0.0  # segmenter rate -- can be well below fps


@dataclass
class LatestFrame:
    """Snapshot of the three views the plan asks to keep visible, plus the
    raw 17x9 grid that is the pipeline's real output."""

    raw_rgb: np.ndarray | None = None       # oriented/fitted preview frame, RGB
    mask_rgb: np.ndarray | None = None      # 2-color mask at lores res, RGB
    grid_rgb: np.ndarray | None = None      # (17, 9, 3) uint8, the display payload
    crop_rect: tuple[int, int, int, int] | None = None
    timings: StageTimings = field(default_factory=StageTimings)
    frame_no: int = 0


class _Slot:
    """Thread-safe single-value mailbox with a monotonic version counter, so
    a reader can either grab whatever's latest or block until something new
    arrives. Used for both the raw-frame and mask handoffs between threads."""

    def __init__(self):
        self._cond = threading.Condition()
        self._value = None
        self._version = 0

    def put(self, value) -> None:
        with self._cond:
            self._value = value
            self._version += 1
            self._cond.notify_all()

    def get(self):
        with self._cond:
            return self._value, self._version

    def wait_for_new(self, last_version: int, timeout: float = 1.0):
        with self._cond:
            self._cond.wait_for(lambda: self._version != last_version, timeout=timeout)
            return self._value, self._version


class LatestFrameHolder:
    """Thread-safe publish/wait point for the composite thread's output.
    Server handlers block on `wait_next()` so a slow HTTP client can't stall
    the pipeline -- they simply miss frames rather than back-pressuring it."""

    def __init__(self):
        self._cond = threading.Condition()
        self._latest: LatestFrame = LatestFrame()

    def publish(self, frame: LatestFrame) -> None:
        with self._cond:
            self._latest = frame
            self._cond.notify_all()

    def get(self) -> LatestFrame:
        with self._cond:
            return self._latest

    def wait_next(self, last_frame_no: int, timeout: float = 1.0) -> LatestFrame:
        with self._cond:
            self._cond.wait_for(
                lambda: self._latest.frame_no != last_frame_no, timeout=timeout
            )
            return self._latest


class Pipeline:
    def __init__(self, live_config: LiveConfig):
        self._live_config = live_config
        self._holder = LatestFrameHolder()
        self._raw_slot = _Slot()   # (main_rgb, lores_gray, frame_no)
        self._mask_slot = _Slot()  # (coverage_full float32, source_frame_no)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._smoothed_coverage: np.ndarray | None = None

    @property
    def holder(self) -> LatestFrameHolder:
        return self._holder

    def latest_grid(self) -> np.ndarray | None:
        """The pipeline's real output for the connectivity/sender code: a
        (17, 9, 3) uint8 array, or None before the first frame."""
        return self._holder.get().grid_rgb

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._run_capture, name="greenpro-capture", daemon=True),
            threading.Thread(target=self._run_inference, name="greenpro-inference", daemon=True),
            threading.Thread(target=self._run_composite, name="greenpro-composite", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)

    # -- capture: reads frames as fast as the source allows -----------------
    def _run_capture(self) -> None:
        cfg = self._live_config.get()
        source = capture.build_source(cfg.capture)
        applied_source = cfg.capture.source
        frame_no = 0
        try:
            while not self._stop.is_set():
                cfg = self._live_config.get()
                if cfg.capture.source != applied_source:
                    source.close()
                    source = capture.build_source(cfg.capture)
                    applied_source = cfg.capture.source

                t0 = time.monotonic()
                try:
                    main_rgb, lores_gray = source.read()
                except StopIteration:
                    break
                # Orientation (rotation/flip) is applied here, immediately on
                # capture -- BEFORE detection/segmentation, not just before
                # display. A person detector expects an upright person; if
                # the camera is physically mounted sideways and rotation is
                # only applied to the output mask afterward, the detector
                # sees a person lying on their side and accuracy suffers.
                # Applying it here means physical mounting orientation is
                # just a config value (cfg.geometry.rotation/hflip/vflip),
                # not something that has to be gotten right with the mount
                # itself. fit/crop (aspect-ratio matching for the display)
                # still happens later, in composite -- that's a framing
                # choice, not a correctness issue for detection.
                main_rgb = geometry.apply_orientation(main_rgb, cfg.geometry)
                lores_gray = geometry.apply_orientation(lores_gray, cfg.geometry)
                capture_time = time.monotonic() - t0

                frame_no += 1
                self._raw_slot.put((main_rgb, lores_gray, frame_no, capture_time))
        finally:
            source.close()

    # -- inference: always processes the latest raw frame, never queues -----
    def _run_inference(self) -> None:
        cfg = self._live_config.get()
        segmenter = segmenters.build_segmenter(cfg.segmenter)
        applied_kind = cfg.segmenter.kind
        last_raw_version = -1
        fps_window_start = time.monotonic()
        fps_window_count = 0
        fps_estimate = 0.0

        while not self._stop.is_set():
            raw, last_raw_version = self._raw_slot.wait_for_new(last_raw_version, timeout=1.0)
            if raw is None:
                continue
            main_rgb, lores_gray, frame_no, _ = raw

            cfg = self._live_config.get()
            if cfg.segmenter.kind != applied_kind:
                segmenter = segmenters.build_segmenter(cfg.segmenter)
                applied_kind = cfg.segmenter.kind
            elif hasattr(segmenter, "update_config"):
                segmenter.update_config(cfg.segmenter.motion)  # type: ignore[attr-defined]

            t0 = time.monotonic()
            coverage_full = segmenter.process(lores_gray, main_rgb)
            segment_time = time.monotonic() - t0

            fps_window_count += 1
            now = time.monotonic()
            if now - fps_window_start >= 1.0:
                fps_estimate = fps_window_count / (now - fps_window_start)
                fps_window_count = 0
                fps_window_start = now

            self._mask_slot.put((coverage_full, frame_no, segment_time, fps_estimate))

    # -- composite: display-facing, paced, free-runs off latest raw + mask --
    def _run_composite(self) -> None:
        frame_no = 0
        fps_window_start = time.monotonic()
        fps_window_count = 0
        fps_estimate = 0.0

        while not self._stop.is_set():
            loop_start = time.monotonic()
            cfg = self._live_config.get()

            raw, _ = self._raw_slot.get()
            mask, _ = self._mask_slot.get()
            if raw is None:
                time.sleep(0.01)
                continue
            main_rgb, lores_gray, _, capture_time = raw
            if mask is not None:
                coverage_full, _, segment_time, inference_fps = mask
            else:
                # No inference result yet (startup). Show an empty mask
                # rather than blocking the display on the first inference pass.
                coverage_full = np.zeros_like(lores_gray, dtype=np.float32)
                segment_time, inference_fps = 0.0, 0.0

            t0 = time.monotonic()
            # coverage_full and main_rgb are already oriented -- rotation/
            # flip was applied once, upstream in _run_capture, before
            # detection/segmentation ever saw the frame (see that method's
            # comment for why). Only fit/crop (aspect-ratio matching for the
            # display) remains to do here.
            fitted_mask = geometry.fit_to_display_aspect(coverage_full, cfg.geometry)
            grid_coverage = geometry.downscale_coverage(fitted_mask)

            # EMA smoothing over the 17x9 grid -- see SmoothingConfig. At 153
            # pixels, per-frame jitter from the neural backends is extremely
            # visible; this is the cheapest, biggest perceived-quality fix.
            alpha = cfg.smoothing.alpha
            if self._smoothed_coverage is None or self._smoothed_coverage.shape != grid_coverage.shape:
                self._smoothed_coverage = grid_coverage.copy()
            else:
                self._smoothed_coverage = (
                    alpha * grid_coverage + (1.0 - alpha) * self._smoothed_coverage
                )
            grid_rgb = geometry.shade_grid(self._smoothed_coverage, cfg.shading)

            fitted_raw = geometry.fit_to_display_aspect(main_rgb, cfg.geometry)
            crop_rect = geometry.crop_overlay_rect(
                main_rgb.shape[1], main_rgb.shape[0], cfg.geometry
            )
            mask_rgb = np.stack([np.clip(coverage_full * 255, 0, 255).astype(np.uint8)] * 3, axis=-1)
            geometry_time = time.monotonic() - t0

            frame_no += 1
            fps_window_count += 1
            now = time.monotonic()
            if now - fps_window_start >= 1.0:
                fps_estimate = fps_window_count / (now - fps_window_start)
                fps_window_count = 0
                fps_window_start = now

            timings = StageTimings(
                capture=capture_time,
                segment=segment_time,
                geometry=geometry_time,
                total=capture_time + segment_time + geometry_time,
                fps=fps_estimate,
                inference_fps=inference_fps,
            )
            self._holder.publish(
                LatestFrame(
                    raw_rgb=fitted_raw,
                    mask_rgb=mask_rgb,
                    grid_rgb=grid_rgb,
                    crop_rect=crop_rect,
                    timings=timings,
                    frame_no=frame_no,
                )
            )

            # Hard pacer: never publish faster than MIN_FRAME_PERIOD,
            # regardless of what cfg.capture.fps claims. This is the
            # independent enforcement of the 30 FPS display ceiling -- see
            # module docstring and config.MAX_DISPLAY_FPS.
            target_period = max(MIN_FRAME_PERIOD, 1.0 / max(1, cfg.capture.fps))
            elapsed = time.monotonic() - loop_start
            remaining = target_period - elapsed
            if remaining > 0:
                time.sleep(remaining)
