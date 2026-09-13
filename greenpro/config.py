"""Configuration for the greenpro pipeline.

All tunables live here and in config.yaml. Config is loaded once at startup
and can be patched live via the server's /config endpoint (see server.py);
every patch goes through `Config.apply_patch`, which re-validates and
re-clamps, so a bad value from the HTTP API can't put the pipeline in an
invalid state.
"""

from __future__ import annotations

import copy
import dataclasses
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Hard ceiling on frames sent to the display. Confirmed by the user and by
# 17x9-Tetris/README.md ("Display.send() should be called *at most* at 30
# FPS"). This is enforced independently in two places: here (config can never
# hold a value above it) and in pipeline.py's frame pacer (a ≥1/30s floor per
# iteration regardless of what fps the config claims). Do not raise this
# without confirming the display can actually take it.
MAX_DISPLAY_FPS = 30

DISPLAY_ROWS = 17
DISPLAY_COLS = 9


def _clamp(value, lo, hi):
    return max(lo, min(value, hi))


@dataclass
class CaptureConfig:
    # "picam" | "image:<path>" | "video:<path>"
    source: str = "picam"
    main_size: tuple[int, int] = (648, 486)
    lores_size: tuple[int, int] = (324, 242)
    fps: int = 20

    def clamp(self) -> None:
        self.fps = _clamp(int(self.fps), 1, MAX_DISPLAY_FPS)


@dataclass
class GeometryConfig:
    rotation: int = 90          # 0 | 90 | 180 | 270
    hflip: bool = False
    vflip: bool = False
    fit: str = "crop"           # "crop" | "squash" | "pad"
    crop_anchor: str = "center"  # "center" | "start" | "end"
    zoom: float = 1.0

    def clamp(self) -> None:
        if self.rotation not in (0, 90, 180, 270):
            self.rotation = 0
        if self.fit not in ("crop", "squash", "pad"):
            self.fit = "crop"
        if self.crop_anchor not in ("center", "start", "end"):
            self.crop_anchor = "center"
        self.zoom = max(1.0, float(self.zoom))


@dataclass
class ShadingConfig:
    fg_color: tuple[int, int, int] = (255, 255, 255)
    bg_color: tuple[int, int, int] = (0, 0, 0)
    shading: str = "linear"     # "linear" | "threshold" | "gamma"
    gamma: float = 1.0
    levels: int = 0             # 0 = no quantization, else N discrete steps
    min_coverage: float = 0.0   # coverage below this -> treated as 0
    threshold: float = 0.5      # used when shading == "threshold"

    def clamp(self) -> None:
        if self.shading not in ("linear", "threshold", "gamma"):
            self.shading = "linear"
        self.gamma = max(0.05, float(self.gamma))
        self.levels = max(0, int(self.levels))
        self.min_coverage = _clamp(float(self.min_coverage), 0.0, 1.0)
        self.threshold = _clamp(float(self.threshold), 0.0, 1.0)
        self.fg_color = tuple(_clamp(int(c), 0, 255) for c in self.fg_color)
        self.bg_color = tuple(_clamp(int(c), 0, 255) for c in self.bg_color)


@dataclass
class MotionConfig:
    history: int = 300
    var_threshold: float = 16.0
    learning_rate: float = -1.0   # -1 => let OpenCV pick automatically
    open_kernel: int = 3
    close_kernel: int = 7
    min_blob_area: int = 40       # in lores pixels

    def clamp(self) -> None:
        self.history = max(1, int(self.history))
        self.var_threshold = max(1.0, float(self.var_threshold))
        self.open_kernel = max(1, int(self.open_kernel) | 1)   # force odd
        self.close_kernel = max(1, int(self.close_kernel) | 1)
        self.min_blob_area = max(0, int(self.min_blob_area))


@dataclass
class NeuralConfig:
    model_path: str = "models/selfie_segmentation.onnx"
    input_size: tuple[int, int] = (256, 256)
    threshold: float = 0.5
    num_threads: int = 4

    def clamp(self) -> None:
        self.threshold = _clamp(float(self.threshold), 0.0, 1.0)
        self.num_threads = max(1, int(self.num_threads))


@dataclass
class TopDownConfig:
    # Cap on how many detected people get a full PP-HumanSeg pass; anyone
    # beyond this (largest boxes served first) falls back to a soft ellipse
    # fill so a crowd degrades gracefully instead of stalling the pipeline.
    # See docs/models.md for the measured per-person cost this trades off.
    max_crops: int = 2
    # Fractional padding added around each detected box before cropping for
    # segmentation, so the crop isn't tight against the person's silhouette.
    crop_padding: float = 0.15
    # PP-HumanSeg is trained on webcam/teleconferencing footage (chest-up
    # framing) -- confirmed live: even given a correct full-body detector
    # box, it under-segments legs, cutting the mask off around the torso.
    # floor_fill blends in a low-confidence soft ellipse spanning the FULL
    # detector box underneath PP-HumanSeg's own mask (via max()), so where
    # PP-HumanSeg is confident (torso/arms/head) that detail wins, and where
    # it isn't (typically legs) the person doesn't just vanish -- they show
    # dimmer instead. 0 disables this and uses PP-HumanSeg's mask as-is.
    floor_fill: float = 0.35

    def clamp(self) -> None:
        self.max_crops = max(1, int(self.max_crops))
        self.crop_padding = _clamp(float(self.crop_padding), 0.0, 1.0)
        self.floor_fill = _clamp(float(self.floor_fill), 0.0, 1.0)


@dataclass
class SmoothingConfig:
    # EMA smoothing over the 17x9 coverage grid, applied before shading.
    # At 153 pixels a single flickering cell is 1/153rd of the whole image,
    # so smoothing the coverage the neural backends produce is worth more
    # than it would be on a normal-resolution display. alpha is the weight
    # given to the *new* frame each update (1.0 = no smoothing).
    alpha: float = 0.6

    def clamp(self) -> None:
        self.alpha = _clamp(float(self.alpha), 0.01, 1.0)


@dataclass
class SegmenterConfig:
    kind: str = "topdown"        # "motion" | "neural" | "boxfill" | "topdown" | "combo"
    combine: str = "max"         # "max" | "min" | "mean" | "motion_gated" (combo only)
    # "nanodet" (default): true full-body COCO boxes, head-to-feet.
    # "mediapipe": faster (measured ~2x, docs/models.md) but its raw box is
    #   NOT a full-body box -- it's an SSD anchor-scale detection box
    #   centered near the face, meant to feed MediaPipe's own separate ROI
    #   calculation from 4 auxiliary landmarks (which detectors.py does not
    #   implement). Confirmed live: it tracked as a square centered on the
    #   face/head, cutting off the body below the chest. Don't default to
    #   this without implementing that ROI step.
    detector_kind: str = "nanodet"
    motion: MotionConfig = field(default_factory=MotionConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    topdown: TopDownConfig = field(default_factory=TopDownConfig)

    def clamp(self) -> None:
        if self.kind not in ("motion", "neural", "boxfill", "topdown", "combo"):
            self.kind = "motion"
        if self.combine not in ("max", "min", "mean", "motion_gated"):
            self.combine = "max"
        if self.detector_kind not in ("mediapipe", "nanodet"):
            self.detector_kind = "mediapipe"
        self.motion.clamp()
        self.neural.clamp()
        self.topdown.clamp()


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    jpeg_quality: int = 70

    def clamp(self) -> None:
        self.jpeg_quality = _clamp(int(self.jpeg_quality), 1, 100)


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    shading: ShadingConfig = field(default_factory=ShadingConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def clamp(self) -> None:
        self.capture.clamp()
        self.geometry.clamp()
        self.shading.clamp()
        self.segmenter.clamp()
        self.smoothing.clamp()
        self.server.clamp()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _merge_dataclass(instance, data: dict) -> None:
    """Recursively apply a dict of overrides onto a dataclass instance,
    ignoring unknown keys rather than raising -- an old client sending a
    stale key should not be able to crash the pipeline."""
    if not data:
        return
    field_names = {f.name: f for f in fields(instance)}
    for key, value in data.items():
        if key not in field_names:
            continue
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)


def load(path: str | Path | None) -> Config:
    """Load a Config from a YAML file, falling back to defaults for anything
    missing or if the file doesn't exist."""
    cfg = Config()
    if path is not None and Path(path).exists():
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        _merge_dataclass(cfg, data)
    cfg.clamp()
    return cfg


class LiveConfig:
    """Thread-safe holder for a Config that can be read and patched while the
    pipeline runs. `get()` returns a deep copy so callers never mutate shared
    state; `apply_patch()` merges + re-clamps atomically under a lock.
    """

    def __init__(self, initial: Config):
        self._lock = threading.Lock()
        self._cfg = initial

    def get(self) -> Config:
        with self._lock:
            return copy.deepcopy(self._cfg)

    def apply_patch(self, patch: dict) -> Config:
        with self._lock:
            _merge_dataclass(self._cfg, patch)
            self._cfg.clamp()
            return copy.deepcopy(self._cfg)
