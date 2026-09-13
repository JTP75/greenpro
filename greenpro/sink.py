"""Network delivery: POST the latest 17x9 grid to Will's Green Building
simulator (see docs/simulator.md for the wire protocol this implements).

Uses stdlib `urllib.request` rather than `requests` -- the Pi's venv is
`--system-site-packages` with only onnxruntime/pyyaml pip-installed (see
docs/setup.md), so a new dependency has a real cost for no benefit here.
Each send opens a fresh HTTPS connection rather than reusing one via
http.client; at the 1-30 req/s this project sends, TLS handshake overhead
is a few ms and not worth the extra connection-lifecycle state -- revisit
only if profiling on the Pi says otherwise.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from .config import LiveConfig
from .pipeline import MIN_FRAME_PERIOD, LatestFrameHolder

log = logging.getLogger(__name__)


class SinkError(Exception):
    """A frame POST failed. `status` is the HTTP status code, or None for a
    connection-level failure (DNS, timeout, refused, etc)."""

    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


class WebDisplaySink:
    """One instance's endpoint: POST https://<base_url>/i/<instance>/frame."""

    def __init__(self, base_url: str, instance: str, timeout: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.instance = instance
        self.timeout = timeout

    @property
    def url(self) -> str:
        return f"{self.base_url}/i/{self.instance}/frame"

    def send(self, grid: np.ndarray) -> int:
        """POST grid (17, 9, 3) uint8 as nested JSON. Returns the HTTP status
        on success (204 per docs/simulator.md); raises SinkError otherwise."""
        body = json.dumps(grid.tolist()).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                # Cloudflare's bot protection in front of the simulator
                # rejects the default "Python-urllib/x.y" UA with a 403
                # (Cloudflare error 1010) -- confirmed by probing. Any
                # ordinary-looking UA clears it.
                "User-Agent": "greenpro-sender/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()  # drain -- body is empty on 204, but be safe
                return resp.status
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise SinkError(exc.code, detail or exc.reason) from exc
        except urllib.error.URLError as exc:
            raise SinkError(None, str(exc.reason)) from exc


class SinkStats:
    """Thread-safe counters for /sink.json and the preview page."""

    def __init__(self):
        self._lock = threading.Lock()
        self.enabled = False
        self.target = ""
        self.sent = 0
        self.failed = 0
        self.consecutive_failures = 0
        self.last_status: int | None = None
        self.last_error: str | None = None
        self.last_latency_ms: float | None = None
        self.send_fps = 0.0
        self._fps_window_start = time.monotonic()
        self._fps_window_count = 0

    def _tick_locked(self) -> None:
        self._fps_window_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self.send_fps = round(self._fps_window_count / elapsed, 2)
            self._fps_window_count = 0
            self._fps_window_start = now

    def record_success(self, status: int, latency_ms: float) -> None:
        with self._lock:
            self.sent += 1
            self.consecutive_failures = 0
            self.last_status = status
            self.last_error = None
            self.last_latency_ms = round(latency_ms, 1)
            self._tick_locked()

    def record_failure(self, status: int | None, error: str) -> None:
        with self._lock:
            self.failed += 1
            self.consecutive_failures += 1
            self.last_status = status
            self.last_error = error
            self._tick_locked()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "target": self.target,
                "sent": self.sent,
                "failed": self.failed,
                "consecutive_failures": self.consecutive_failures,
                "last_status": self.last_status,
                "last_error": self.last_error,
                "last_latency_ms": self.last_latency_ms,
                "send_fps": self.send_fps,
            }


class FrameSender:
    """Background thread: takes each new grid from a Pipeline- or
    PatternPipeline-shaped `.holder` and POSTs it via WebDisplaySink. Mirrors
    Pipeline's start()/stop() shape so run.py treats it identically.

    Re-reads live_config.get() every iteration so sink.enabled/instance/
    base_url/fps can be changed live via POST /config with no restart.
    A send failure is recorded in stats and the loop backs off (capped),
    but the thread never dies and the source pipeline is never touched --
    one bad instance name or a wifi blip can't take down capture.
    """

    MAX_BACKOFF = 10.0

    def __init__(self, holder: LatestFrameHolder, live_config: LiveConfig):
        self._holder = holder
        self._live_config = live_config
        self._stats = SinkStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def stats(self) -> SinkStats:
        return self._stats

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="greenpro-sender", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        last_frame_no = -1
        backoff = 0.0

        while not self._stop.is_set():
            cfg = self._live_config.get().sink
            self._stats.enabled = cfg.enabled
            self._stats.target = f"{cfg.base_url}/i/{cfg.instance}/frame" if cfg.instance else ""

            if not cfg.enabled or not cfg.instance:
                # Idle without a tight spin loop, but stay responsive to
                # being (re-)enabled or to stop().
                self._stop.wait(timeout=0.5)
                continue

            frame = self._holder.wait_next(last_frame_no, timeout=1.0)
            if frame.frame_no == last_frame_no or frame.grid_rgb is None:
                continue
            last_frame_no = frame.frame_no

            if backoff > 0:
                if self._stop.wait(timeout=backoff):
                    break  # stop() was called during backoff
                # Config may have changed while backing off (e.g. a live fix
                # to sink.instance) -- a backoff can run up to MAX_BACKOFF
                # seconds, long enough that using the pre-backoff snapshot
                # here would resend to a target the operator already fixed.
                cfg = self._live_config.get().sink
                self._stats.enabled = cfg.enabled
                self._stats.target = f"{cfg.base_url}/i/{cfg.instance}/frame" if cfg.instance else ""
                if not cfg.enabled or not cfg.instance:
                    continue

            sink = WebDisplaySink(cfg.base_url, cfg.instance, timeout=cfg.timeout)
            t0 = time.monotonic()
            try:
                status = sink.send(frame.grid_rgb)
            except SinkError as exc:
                self._stats.record_failure(exc.status, str(exc))
                backoff = min(self.MAX_BACKOFF, backoff * 2 if backoff else cfg.retry_backoff)
                log.warning("send to %s failed: %s", self._stats.target, exc)
                continue

            latency_ms = (time.monotonic() - t0) * 1000
            self._stats.record_success(status, latency_ms)
            backoff = 0.0

            # Our own pacing floor on top of wait_next's cadence -- sink.fps
            # can be set lower than the source pipeline's own rate, and this
            # is always clamped below the hard 30 FPS display ceiling too.
            target_period = max(1.0 / max(1, cfg.fps), MIN_FRAME_PERIOD)
            remaining = target_period - (time.monotonic() - t0)
            if remaining > 0:
                self._stop.wait(timeout=remaining)
