"""Frame sources.

Every source yields (main_rgb, lores_gray) pairs of numpy arrays:
  main_rgb:   (H, W, 3) uint8, RGB order -- used for the raw preview and as
              input to the neural segmenter.
  lores_gray: (h, w) uint8 grayscale -- used for the motion segmenter and as
              the base resolution geometry/downscale operates on.

PiCameraSource pulls both from a single picamera2 request so they represent
the same instant. ImageSource/VideoSource exist so the rest of the pipeline
(geometry, segmenters, server) can be developed and tested without a camera
attached -- point --source at a still or a clip captured on the Pi.
"""

from __future__ import annotations

import abc
import time
from pathlib import Path

import cv2
import numpy as np

from .config import CaptureConfig


class Source(abc.ABC):
    @abc.abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the next (main_rgb, lores_gray) pair. Blocks until a frame
        is ready; raises StopIteration when the source is exhausted (never,
        for a live camera; once per loop, for a file source)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class PiCameraSource(Source):
    """Reads dual streams off a Raspberry Pi camera via picamera2.

    `main` is RGB888 at `main_size` (used for preview + neural input).
    `lores` is YUV420 at `lores_size`, hardware-scaled by the ISP for free;
    we take the Y plane directly as grayscale rather than converting RGB,
    which is the cheapest possible way to get a segmenter input frame.

    Actual negotiated sizes can differ slightly from what was requested
    (e.g. odd dimensions get rounded), so the real sizes are read back from
    the camera's configuration after `configure()` rather than assumed.
    """

    def __init__(self, cfg: CaptureConfig):
        from picamera2 import Picamera2  # imported lazily: not present off-Pi

        self._picam2 = Picamera2()
        video_config = self._picam2.create_video_configuration(
            main={"size": tuple(cfg.main_size), "format": "RGB888"},
            lores={"size": tuple(cfg.lores_size), "format": "YUV420"},
            controls={"FrameRate": cfg.fps},
        )
        self._picam2.configure(video_config)
        lw, lh = self._picam2.camera_configuration()["lores"]["size"]
        self._lores_w, self._lores_h = lw, lh
        self._picam2.start()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        request = self._picam2.capture_request()
        try:
            main = request.make_array("main")          # (H, W, 3), RGB
            yuv = request.make_array("lores")           # (h*3/2, w), YUV420
            gray = yuv[: self._lores_h, : self._lores_w]
            return main, np.ascontiguousarray(gray)
        finally:
            request.release()

    def close(self) -> None:
        self._picam2.stop()
        self._picam2.close()


class _LoopingFrameSource(Source):
    """Shared logic for image/video sources: derive lores_gray from
    main_rgb via a plain resize + grayscale convert (no ISP available off a
    real sensor), and pace reads to roughly `fps` so downstream fps-dependent
    logic (e.g. MOG2 learning rate) behaves similarly to live capture."""

    def __init__(self, lores_size: tuple[int, int], fps: int):
        self._lores_w, self._lores_h = lores_size
        self._period = 1.0 / max(1, fps)
        self._last_read = 0.0

    def _pace(self) -> None:
        now = time.monotonic()
        wait = self._period - (now - self._last_read)
        if wait > 0:
            time.sleep(wait)
        self._last_read = time.monotonic()

    def _derive_lores(self, main_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(
            gray, (self._lores_w, self._lores_h), interpolation=cv2.INTER_AREA
        )


class ImageSource(_LoopingFrameSource):
    """Replays a single still image forever. Useful for tuning geometry and
    shading without any motion to worry about."""

    def __init__(self, path: str | Path, cfg: CaptureConfig):
        super().__init__(tuple(cfg.lores_size), cfg.fps)
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        img_bgr = cv2.resize(img_bgr, tuple(cfg.main_size))
        self._main_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        self._pace()
        return self._main_rgb.copy(), self._derive_lores(self._main_rgb)


class VideoSource(_LoopingFrameSource):
    """Replays a video file on loop via OpenCV's VideoCapture."""

    def __init__(self, path: str | Path, cfg: CaptureConfig):
        super().__init__(tuple(cfg.lores_size), cfg.fps)
        self._path = str(path)
        self._main_size = tuple(cfg.main_size)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"could not open video: {path}")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        self._pace()
        ok, frame_bgr = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = self._cap.read()
            if not ok:
                raise StopIteration("video source exhausted and could not loop")
        frame_bgr = cv2.resize(frame_bgr, self._main_size)
        main_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return main_rgb, self._derive_lores(main_rgb)

    def close(self) -> None:
        self._cap.release()


def build_source(cfg: CaptureConfig) -> Source:
    """Factory: cfg.source is "picam", "image:<path>", or "video:<path>"."""
    spec = cfg.source
    if spec == "picam":
        return PiCameraSource(cfg)
    if spec.startswith("image:"):
        return ImageSource(spec.split(":", 1)[1], cfg)
    if spec.startswith("video:"):
        return VideoSource(spec.split(":", 1)[1], cfg)
    raise ValueError(f"unknown capture source: {spec!r}")
