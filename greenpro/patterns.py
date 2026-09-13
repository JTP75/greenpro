"""Synthetic 17x9 grid sources, for exercising the network-delivery path
without a camera, a segmenter, or the Pi.

The capture/segment/geometry chain in pipeline.py is deliberately bypassed
here, not reused: geometry.apply_orientation and fit_to_display_aspect exist
to correct a *camera's* framing (rotation, crop, aspect), and would rotate
or crop a hand-crafted pattern, destroying the exact pixel-to-cell mapping
these patterns exist to verify. Instead, `PatternPipeline` publishes
finished (17, 9, 3) grids straight into the same `LatestFrameHolder` the
real `Pipeline` uses, so everything downstream -- the preview server,
/frame.json, and the sender in sink.py -- consumes synthetic and real
frames through an identical interface.

Selected via the existing --source flag, following the "image:<path>" /
"video:<path>" convention capture.build_source already parses:

    python run.py --source pattern:blink
    python run.py --source pattern:corners
    python run.py --source pattern:solid:255,0,0
"""

from __future__ import annotations

import colorsys
import logging
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .config import DISPLAY_COLS, DISPLAY_ROWS, LiveConfig
from .pipeline import MIN_FRAME_PERIOD, LatestFrame, LatestFrameHolder, StageTimings

log = logging.getLogger(__name__)

Renderer = Callable[[int, float], np.ndarray]

# How fast the pattern loop generates frames. Independent of cfg.capture.fps
# (there's no capture), but still paced below MIN_FRAME_PERIOD by _run().
DEFAULT_PATTERN_FPS = 10.0

# Nearest-neighbor upscale factor for the raw/mask preview panes, so
# /raw.mjpg and /mask.mjpg stay coherent while a PatternPipeline is active.
PREVIEW_UPSCALE = 24


def _blank() -> np.ndarray:
    return np.zeros((DISPLAY_ROWS, DISPLAY_COLS, 3), dtype=np.uint8)


def blink(frame_no: int, t: float) -> np.ndarray:
    """Two pixels blinking ~1 Hz. The primary "did anything arrive" signal."""
    grid = _blank()
    if int(t * 2) % 2 == 0:  # on for 0.5s, off for 0.5s
        grid[0, 0] = (255, 60, 60)
        grid[DISPLAY_ROWS - 1, DISPLAY_COLS - 1] = (60, 140, 255)
    return grid


def corners(frame_no: int, t: float) -> np.ndarray:
    """A distinct color in each corner, plus a marker one row down from the
    top-left, so watching the real render identifies which physical cell is
    grid[0][0] and confirms the top-to-bottom direction."""
    grid = _blank()
    grid[0, 0] = (255, 0, 0)  # top-left
    grid[0, DISPLAY_COLS - 1] = (0, 255, 0)  # top-right
    grid[DISPLAY_ROWS - 1, 0] = (0, 0, 255)  # bottom-left
    grid[DISPLAY_ROWS - 1, DISPLAY_COLS - 1] = (255, 255, 0)  # bottom-right
    grid[1, 0] = (255, 255, 255)  # one row below top-left: confirms row direction
    return grid


def bar(frame_no: int, t: float) -> np.ndarray:
    """One lit row bouncing top<->bottom. Shows motion/continuity and makes
    dropped frames visible as a stutter."""
    grid = _blank()
    period = 2.0  # seconds for a full down-then-up sweep
    phase = (t % period) / period  # 0..1
    triangle = 1.0 - abs(2.0 * phase - 1.0)  # 0 -> 1 -> 0
    row = round(triangle * (DISPLAY_ROWS - 1))
    grid[row, :] = (255, 255, 255)
    return grid


def rainbow(frame_no: int, t: float) -> np.ndarray:
    """Hue drifting down the rows over time. Exercises the full-color path;
    all 153 cells are non-zero."""
    grid = _blank()
    for row in range(DISPLAY_ROWS):
        hue = ((row / DISPLAY_ROWS) + 0.15 * t) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        grid[row, :] = (int(r * 255), int(g * 255), int(b * 255))
    return grid


def solid(frame_no: int, t: float, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """One color everywhere -- the easiest thing to eyeball against the real
    render's white balance/gamma."""
    grid = _blank()
    grid[:, :] = color
    return grid


_RENDERERS: dict[str, Renderer] = {
    "blink": blink,
    "corners": corners,
    "bar": bar,
    "rainbow": rainbow,
    "solid": solid,
}


def build_renderer(spec: str) -> Renderer:
    """spec is the part after "pattern:", e.g. "blink" or "solid:255,0,0"."""
    name, _, arg = spec.partition(":")
    if name not in _RENDERERS:
        raise ValueError(
            f"unknown pattern: {name!r} (known: {', '.join(sorted(_RENDERERS))})"
        )
    if name == "solid" and arg:
        try:
            r, g, b = (int(v) for v in arg.split(","))
        except ValueError as exc:
            raise ValueError(f"pattern:solid:<r,g,b> expected, got {arg!r}") from exc
        return lambda frame_no, t: solid(frame_no, t, color=(r, g, b))
    return _RENDERERS[name]


def _preview_frame(grid: np.ndarray) -> np.ndarray:
    return cv2.resize(
        grid,
        (DISPLAY_COLS * PREVIEW_UPSCALE, DISPLAY_ROWS * PREVIEW_UPSCALE),
        interpolation=cv2.INTER_NEAREST,
    )


class PatternPipeline:
    """Duck-typed sibling of pipeline.Pipeline: same .holder / .latest_grid()
    / .start() / .stop() surface, so run.py, server.py and sink.py don't need
    to know whether frames are coming from a camera or a synthetic pattern."""

    def __init__(self, live_config: LiveConfig, render: Renderer, fps: float = DEFAULT_PATTERN_FPS):
        self._live_config = live_config  # unused today; kept for interface parity
        self._render = render
        self._fps = fps
        self._holder = LatestFrameHolder()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def holder(self) -> LatestFrameHolder:
        return self._holder

    def latest_grid(self) -> np.ndarray | None:
        return self._holder.get().grid_rgb

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="greenpro-pattern", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        frame_no = 0
        t_start = time.monotonic()
        fps_window_start = t_start
        fps_window_count = 0
        fps_estimate = 0.0

        while not self._stop.is_set():
            loop_start = time.monotonic()
            t = loop_start - t_start

            grid = self._render(frame_no, t)
            preview = _preview_frame(grid)

            frame_no += 1
            fps_window_count += 1
            now = time.monotonic()
            if now - fps_window_start >= 1.0:
                fps_estimate = fps_window_count / (now - fps_window_start)
                fps_window_count = 0
                fps_window_start = now

            self._holder.publish(
                LatestFrame(
                    raw_rgb=preview,
                    mask_rgb=preview,
                    grid_rgb=grid,
                    crop_rect=None,
                    timings=StageTimings(fps=fps_estimate),
                    frame_no=frame_no,
                )
            )

            # Same independent pacing floor pipeline.py uses -- a pattern
            # source must not be able to drive the display faster than the
            # 30 FPS ceiling either.
            target_period = max(MIN_FRAME_PERIOD, 1.0 / max(1, self._fps))
            elapsed = time.monotonic() - loop_start
            remaining = target_period - elapsed
            if remaining > 0:
                time.sleep(remaining)
