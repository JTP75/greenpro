"""Human-figure segmentation backends.

Every Segmenter.process() returns a float32 array in [0, 1], same H x W as
the input grayscale frame -- 1 where a human figure is judged present, 0
where not, with fractional values allowed (and encouraged: they feed the
area-average downscale smoothly and let ComboSegmenter blend backends
usefully instead of just OR-ing two binary masks).
"""

from __future__ import annotations

import abc
import logging

import cv2
import numpy as np

from .config import MotionConfig, NeuralConfig, SegmenterConfig, TopDownConfig
from .detectors import Box, PersonDetector, build_detector
from .vendor_loader import load_module, model_weights_path

log = logging.getLogger(__name__)


class Segmenter(abc.ABC):
    @abc.abstractmethod
    def process(self, gray: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
        """gray: (h, w) uint8 lores grayscale. rgb: (H, W, 3) uint8 full-res
        RGB if available (neural backends want this for a sharper input;
        motion-only backends can ignore it). Returns float32 (h, w) in [0,1]."""
        raise NotImplementedError


class MotionSegmenter(Segmenter):
    """Background subtraction via MOG2. Cheap, no model file, works the
    instant something moves. Loses track of a subject who stops moving
    (they get absorbed into the learned background), and reacts to any
    motion, not just people -- doors, shadows, lighting changes.
    """

    def __init__(self, cfg: MotionConfig):
        self._cfg = cfg
        self._build_subtractor(cfg)

    def _build_subtractor(self, cfg: MotionConfig) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history,
            varThreshold=cfg.var_threshold,
            detectShadows=True,
        )
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel)
        )
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel)
        )

    def update_config(self, cfg: MotionConfig) -> None:
        """Rebuild the subtractor if its own params changed; kernels can be
        swapped without losing background history."""
        if (
            cfg.history != self._cfg.history
            or cfg.var_threshold != self._cfg.var_threshold
        ):
            self._build_subtractor(cfg)
        else:
            self._open_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel)
            )
            self._close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel)
            )
        self._cfg = cfg

    def process(self, gray: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
        fg = self._subtractor.apply(gray, learningRate=self._cfg.learning_rate)
        # MOG2 with detectShadows marks shadow pixels as 127; keep only
        # confident foreground (255), not shadows.
        binary = (fg == 255).astype(np.uint8) * 255

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._close_kernel)

        if self._cfg.min_blob_area > 0:
            binary = self._drop_small_blobs(binary, self._cfg.min_blob_area)

        return (binary.astype(np.float32)) / 255.0

    @staticmethod
    def _drop_small_blobs(binary: np.ndarray, min_area: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        out = np.zeros_like(binary)
        for label in range(1, num_labels):  # 0 is background
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                out[labels == label] = 255
        return out


class NeuralSegmenter(Segmenter):
    """Person segmentation via an ONNX model (e.g. MediaPipe Selfie
    Segmentation exported to ONNX). Holds a still subject; ignores
    non-human motion. Needs onnxruntime and a model file -- see
    docs/models.md for where to get one.

    Contingency (see plan): if the selfie model performs poorly on small,
    distant, full-body figures, swap in a person-detector model that fills
    bounding boxes instead of tracing a silhouette -- at 9x17 output
    resolution the two are nearly indistinguishable. That only requires a
    new Segmenter subclass; nothing else in the pipeline changes.
    """

    def __init__(self, cfg: NeuralConfig):
        import onnxruntime as ort

        self._cfg = cfg
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.num_threads
        self._session = ort.InferenceSession(
            cfg.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def process(self, gray: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
        h, w = gray.shape[:2]
        if rgb is None:
            # No full-res frame available (shouldn't happen in normal
            # operation); fall back to upsampling grayscale to 3 channels.
            source = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            source = rgb

        iw, ih = self._cfg.input_size
        resized = cv2.resize(source, (iw, ih), interpolation=cv2.INTER_LINEAR)
        blob = (resized.astype(np.float32) / 255.0)[None, ...]  # NHWC

        output = self._session.run([self._output_name], {self._input_name: blob})[0]
        prob = np.squeeze(output)
        if prob.ndim == 3:
            # some export variants emit (H, W, C) with C=1 or 2 classes
            prob = prob[..., -1] if prob.shape[-1] <= 2 else prob[0]

        prob = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        prob = np.clip(prob, 0.0, 1.0)

        if self._cfg.threshold > 0:
            prob = np.where(prob >= self._cfg.threshold, prob, 0.0)
        return prob


def _draw_soft_ellipse(mask: np.ndarray, box: Box, amplitude: float = 1.0) -> None:
    """Fills box with a soft (anti-aliased, vertically stretched) ellipse
    rather than a hard rectangle. A rectangle's corners survive the 17x9
    area-average as visible squared-off blocks; an ellipse reads as a person
    once downscaled. Drawn in two passes (filled + one blurred edge) for a
    soft boundary rather than a binary one, matching the coverage semantics
    every other Segmenter returns. `amplitude` < 1.0 draws a dimmer "guess"
    layer -- see TopDownConfig.floor_fill."""
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    axes = (max(1, int(box.w * 0.42)), max(1, int(box.h * 0.48)))
    layer = np.zeros_like(mask)
    cv2.ellipse(layer, (int(cx), int(cy)), axes, 0, 0, 360, amplitude, thickness=-1)
    blur_k = max(3, (min(axes) // 4) | 1)  # odd kernel, scaled to box size
    layer = cv2.GaussianBlur(layer, (blur_k, blur_k), 0)
    np.maximum(mask, layer, out=mask)


class BoxFillSegmenter(Segmenter):
    """Detector only, no segmentation model -- each detected person becomes a
    soft-edged filled ellipse. This is the fast path: about as fast as the
    detector alone (~20fps measured for MediaPipe on Pi 5, see
    docs/models.md), and at 9x17 output resolution a filled silhouette shape
    is nearly indistinguishable from a traced one. Good default if
    TopDownSegmenter's extra per-person segmentation pass is too slow for a
    given scene (many people at once).
    """

    def __init__(self, detector_kind: str = "mediapipe"):
        self._detector: PersonDetector = build_detector(detector_kind)

    def process(self, gray: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
        h, w = gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        if rgb is None:
            return mask
        scale_x, scale_y = w / rgb.shape[1], h / rgb.shape[0]
        for box in self._detector.detect(rgb):
            scaled = Box(
                int(box.x * scale_x), int(box.y * scale_y),
                int(box.w * scale_x), int(box.h * scale_y), box.score,
            ).clipped(w, h)
            if scaled.w > 0 and scaled.h > 0:
                _draw_soft_ellipse(mask, scaled)
        return mask


class TopDownSegmenter(Segmenter):
    """The room-scale answer to a portrait-trained segmentation model: run a
    full-body person detector over the whole frame (any distance), then run
    PP-HumanSeg on each detected person's *cropped* box. A tight per-person
    crop resized to PP-HumanSeg's 192x192 input is exactly the close-up
    framing it was trained on -- see docs/models.md for why this combination
    was chosen over every single-model alternative that was tried.

    Detected boxes are capped at `max_crops` (largest-area first) to bound
    worst-case cost in a crowd; any boxes beyond that fall back to the same
    soft-ellipse fill BoxFillSegmenter uses, so a crowd degrades gracefully
    instead of the frame rate collapsing.
    """

    def __init__(self, cfg: TopDownConfig, detector_kind: str = "mediapipe"):
        self._cfg = cfg
        self._detector: PersonDetector = build_detector(detector_kind)
        pphumanseg_mod = load_module("human_segmentation_pphumanseg/pphumanseg.py")
        # int8bq fails to load under this OpenCV build's ONNX importer
        # (DequantizeLinear parse error -- see docs/models.md); float32
        # measured at ~30fps on a single crop on Pi 5, plenty fast.
        weights = model_weights_path(
            "human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx"
        )
        self._segmenter = pphumanseg_mod.PPHumanSeg(modelPath=weights)

    def process(self, gray: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
        h, w = gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        if rgb is None:
            return mask

        rgb_h, rgb_w = rgb.shape[:2]
        scale_x, scale_y = w / rgb_w, h / rgb_h
        pad = self._cfg.crop_padding

        boxes = sorted(self._detector.detect(rgb), key=lambda b: b.area(), reverse=True)
        segmented, overflow = boxes[: self._cfg.max_crops], boxes[self._cfg.max_crops :]

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for box in segmented:
            pad_w, pad_h = int(box.w * pad), int(box.h * pad)
            padded = Box(
                box.x - pad_w, box.y - pad_h, box.w + 2 * pad_w, box.h + 2 * pad_h, box.score
            ).clipped(rgb_w, rgb_h)
            if padded.w <= 0 or padded.h <= 0:
                continue
            crop = bgr[padded.y : padded.y + padded.h, padded.x : padded.x + padded.w]
            person_mask = self._segmenter.infer(crop)[0].astype(np.float32)  # (h, w) in {0,1}

            scaled_box = Box(
                int(padded.x * scale_x), int(padded.y * scale_y),
                int(padded.w * scale_x), int(padded.h * scale_y), box.score,
            ).clipped(w, h)
            if scaled_box.w <= 0 or scaled_box.h <= 0:
                continue
            resized = cv2.resize(
                person_mask, (scaled_box.w, scaled_box.h), interpolation=cv2.INTER_LINEAR
            )
            region = mask[scaled_box.y : scaled_box.y + scaled_box.h, scaled_box.x : scaled_box.x + scaled_box.w]
            np.maximum(region, resized, out=region)

        for box in overflow:
            scaled = Box(
                int(box.x * scale_x), int(box.y * scale_y),
                int(box.w * scale_x), int(box.h * scale_y), box.score,
            ).clipped(w, h)
            if scaled.w > 0 and scaled.h > 0:
                _draw_soft_ellipse(mask, scaled)

        return mask


class ComboSegmenter(Segmenter):
    """Runs motion and neural segmenters together. `combine` controls how
    their outputs merge:
      max          -- union (default): neural holds a still person, motion
                      catches anyone the model misses.
      min          -- intersection: only where both agree.
      mean         -- average of the two.
      motion_gated -- neural's mask, but zeroed wherever there's no motion
                      at all (rejects static false positives from the model).
    """

    def __init__(self, cfg: SegmenterConfig):
        self._cfg = cfg
        self._motion = MotionSegmenter(cfg.motion)
        try:
            self._neural: Segmenter | None = NeuralSegmenter(cfg.neural)
        except Exception:
            log.warning(
                "neural segmenter failed to load; combo falls back to motion-only",
                exc_info=True,
            )
            self._neural = None

    def process(self, gray: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
        motion_mask = self._motion.process(gray, rgb)
        if self._neural is None:
            return motion_mask
        neural_mask = self._neural.process(gray, rgb)

        combine = self._cfg.combine
        if combine == "min":
            return np.minimum(motion_mask, neural_mask)
        if combine == "mean":
            return (motion_mask + neural_mask) / 2.0
        if combine == "motion_gated":
            return np.where(motion_mask > 0, neural_mask, 0.0)
        return np.maximum(motion_mask, neural_mask)  # "max" / default


def build_segmenter(cfg: SegmenterConfig) -> Segmenter:
    if cfg.kind == "motion":
        return MotionSegmenter(cfg.motion)
    if cfg.kind == "neural":
        try:
            return NeuralSegmenter(cfg.neural)
        except Exception:
            log.warning(
                "neural segmenter failed to load; falling back to motion", exc_info=True
            )
            return MotionSegmenter(cfg.motion)
    if cfg.kind == "boxfill":
        try:
            return BoxFillSegmenter(cfg.detector_kind)
        except Exception:
            log.warning(
                "boxfill segmenter failed to load; falling back to motion", exc_info=True
            )
            return MotionSegmenter(cfg.motion)
    if cfg.kind == "topdown":
        try:
            return TopDownSegmenter(cfg.topdown, cfg.detector_kind)
        except Exception:
            log.warning(
                "topdown segmenter failed to load; falling back to motion", exc_info=True
            )
            return MotionSegmenter(cfg.motion)
    if cfg.kind == "combo":
        return ComboSegmenter(cfg)
    raise ValueError(f"unknown segmenter kind: {cfg.kind!r}")
