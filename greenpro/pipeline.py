"""Capture -> segment -> geometry/downscale -> publish, on a background
thread. Reads from a LiveConfig so the server can patch parameters (crop,
shading, segmenter choice, ...) while frames keep flowing, and rebuilds the
source/segmenter objects when a change requires it (e.g. switching segmenter
kind, or changing the capture source).

The 30 FPS ceiling on frames sent to the display (see config.MAX_DISPLAY_FPS)
is enforced *here*, independently of whatever fps value config claims: each
loop iteration is paced to a minimum period, so even a bug that let an
out-of-range fps through config would not be able to drive the display
faster than the hardware allows.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import capture, geometry, segmenters
from .config import DISPLAY_COLS, DISPLAY_ROWS, MAX_DISPLAY_FPS, Config, LiveConfig

log = logging.getLogger(__name__)

MIN_FRAME_PERIOD = 1.0 / MAX_DISPLAY_FPS  # hard floor, independent of config


@dataclass
class StageTimings:
    capture: float = 0.0
    segment: float = 0.0
    geometry: float = 0.0
    total: float = 0.0
    fps: float = 0.0


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


class LatestFrameHolder:
    """Thread-safe publish/wait point. Server handlers block on `wait_next()`
    so a slow HTTP client can't stall the capture loop -- they simply miss
    frames rather than back-pressuring the pipeline."""

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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def holder(self) -> LatestFrameHolder:
        return self._holder

    def latest_grid(self) -> np.ndarray | None:
        """The pipeline's real output for the connectivity/sender code that
        will be built later: a (17, 9, 3) uint8 array, or None before the
        first frame."""
        return self._holder.get().grid_rgb

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="greenpro-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        cfg = self._live_config.get()
        source = capture.build_source(cfg.capture)
        segmenter = segmenters.build_segmenter(cfg.segmenter)
        applied_capture_source = cfg.capture.source
        applied_segmenter_kind = cfg.segmenter.kind

        frame_no = 0
        fps_window_start = time.monotonic()
        fps_window_count = 0
        fps_estimate = 0.0

        try:
            while not self._stop.is_set():
                loop_start = time.monotonic()
                cfg = self._live_config.get()

                # Rebuild the source only if its identity actually changed --
                # camera sources are expensive to reopen.
                if cfg.capture.source != applied_capture_source:
                    source.close()
                    source = capture.build_source(cfg.capture)
                    applied_capture_source = cfg.capture.source

                if cfg.segmenter.kind != applied_segmenter_kind:
                    segmenter = segmenters.build_segmenter(cfg.segmenter)
                    applied_segmenter_kind = cfg.segmenter.kind
                elif hasattr(segmenter, "update_config"):
                    segmenter.update_config(cfg.segmenter.motion)  # type: ignore[attr-defined]

                t0 = time.monotonic()
                try:
                    main_rgb, lores_gray = source.read()
                except StopIteration:
                    break
                t1 = time.monotonic()

                coverage_full = segmenter.process(lores_gray, main_rgb)
                t2 = time.monotonic()

                oriented_mask = geometry.apply_orientation(coverage_full, cfg.geometry)
                fitted_mask = geometry.fit_to_display_aspect(oriented_mask, cfg.geometry)
                grid_coverage = geometry.downscale_coverage(fitted_mask)
                grid_rgb = geometry.shade_grid(grid_coverage, cfg.shading)

                oriented_raw = geometry.apply_orientation(main_rgb, cfg.geometry)
                fitted_raw = geometry.fit_to_display_aspect(oriented_raw, cfg.geometry)
                crop_rect = geometry.crop_overlay_rect(
                    oriented_raw.shape[1], oriented_raw.shape[0], cfg.geometry
                )

                mask_rgb = np.stack([np.clip(oriented_mask * 255, 0, 255).astype(np.uint8)] * 3, axis=-1)
                t3 = time.monotonic()

                frame_no += 1
                fps_window_count += 1
                now = time.monotonic()
                if now - fps_window_start >= 1.0:
                    fps_estimate = fps_window_count / (now - fps_window_start)
                    fps_window_count = 0
                    fps_window_start = now

                timings = StageTimings(
                    capture=t1 - t0,
                    segment=t2 - t1,
                    geometry=t3 - t2,
                    total=t3 - t0,
                    fps=fps_estimate,
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

                # Hard pacer: never iterate faster than MIN_FRAME_PERIOD,
                # regardless of what cfg.capture.fps claims. This is the
                # independent enforcement of the 30 FPS display ceiling --
                # see module docstring and config.MAX_DISPLAY_FPS.
                target_period = max(MIN_FRAME_PERIOD, 1.0 / max(1, cfg.capture.fps))
                elapsed = time.monotonic() - loop_start
                remaining = target_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            source.close()
