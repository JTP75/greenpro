"""Person detectors: find where people are in a full frame, independent of
distance. This is the piece that makes a portrait-trained segmentation model
(PP-HumanSeg) usable at room scale -- see segmenters.py's TopDownSegmenter and
docs/models.md for why.

Both backends wrap classes vendored verbatim from opencv/opencv_zoo (see
vendor_loader.py for why they're vendored rather than reimplemented) and
normalize their very different output formats into one `Box` shape.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import cv2
import numpy as np

from .vendor_loader import load_module, model_weights_path


@dataclass
class Box:
    """Full-frame pixel coordinates, top-left origin."""

    x: int
    y: int
    w: int
    h: int
    score: float

    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def clipped(self, frame_w: int, frame_h: int) -> "Box":
        x0 = max(0, self.x)
        y0 = max(0, self.y)
        x1 = min(frame_w, self.x + self.w)
        y1 = min(frame_h, self.y + self.h)
        return Box(x0, y0, max(0, x1 - x0), max(0, y1 - y0), self.score)


class PersonDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, rgb: np.ndarray) -> list[Box]:
        """rgb: (H, W, 3) uint8, full frame. Returns person boxes, any order."""
        raise NotImplementedError


class NanoDetPersonDetector(PersonDetector):
    """COCO object detector (anchor-free, DFL box regression), filtered to
    the 'person' class (COCO index 0). Simple, well-documented decode with
    no external anchor table -- see vendor/opencv_zoo/models/object_detection_nanodet/nanodet.py.

    Input is a fixed 416x416 square; non-square frames are letterboxed
    (aspect-preserving resize + black-bar padding) exactly as the model's own
    demo.py does, and boxes are mapped back out of letterbox space afterward.
    """

    PERSON_CLASS_ID = 0
    INPUT_SIZE = (416, 416)

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = 0.4,
        nms_threshold: float = 0.5,
    ):
        nanodet_mod = load_module("object_detection_nanodet/nanodet.py")
        # int8bq fails to load under this OpenCV build's ONNX importer
        # (DequantizeLinear parse error -- see docs/models.md); float32
        # measured fast enough on Pi 5 that this isn't a real loss.
        weights = model_path or model_weights_path(
            "object_detection_nanodet/object_detection_nanodet_2022nov.onnx"
        )
        self._model = nanodet_mod.NanoDet(
            modelPath=weights, prob_threshold=confidence, iou_threshold=nms_threshold
        )

    @staticmethod
    def _letterbox(img: np.ndarray, target_size=(416, 416)):
        # Ported verbatim from opencv_zoo/models/object_detection_nanodet/demo.py's
        # letterbox()/unletterbox() -- see docs/models.md for why this is
        # kept in sync with the vendored NanoDet class rather than derived
        # independently (the anchor grid is generated for this exact 416x416
        # square, so the resize/pad convention must match precisely).
        top, left, newh, neww = 0, 0, target_size[0], target_size[1]
        h, w = img.shape[:2]
        if h != w:
            hw_scale = h / w
            if hw_scale > 1:
                newh, neww = target_size[0], int(target_size[1] / hw_scale)
                img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((target_size[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img, 0, 0, left, target_size[1] - neww - left, cv2.BORDER_CONSTANT, value=0
                )
            else:
                newh, neww = int(target_size[0] * hw_scale), target_size[1]
                img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((target_size[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img, top, target_size[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0
                )
        else:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img, (top, left, newh, neww)

    @staticmethod
    def _unletterbox(bbox, original_shape, letterbox_scale):
        h, w = original_shape
        top, left, newh, neww = letterbox_scale
        ret = bbox.astype(np.float64).copy()
        if h == w:
            ratio = h / newh
            return ret * ratio
        ratioh, ratiow = h / newh, w / neww
        ret[0] = max((ret[0] - left) * ratiow, 0)
        ret[1] = max((ret[1] - top) * ratioh, 0)
        ret[2] = min((ret[2] - left) * ratiow, w)
        ret[3] = min((ret[3] - top) * ratioh, h)
        return ret

    def detect(self, rgb: np.ndarray) -> list[Box]:
        h, w = rgb.shape[:2]
        letterboxed, scale = self._letterbox(rgb, self.INPUT_SIZE)
        preds = self._model.infer(letterboxed)
        boxes: list[Box] = []
        for pred in preds:
            class_id = int(pred[-1])
            if class_id != self.PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = self._unletterbox(pred[:4], (h, w), scale)
            boxes.append(Box(int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(pred[-2])))
        return boxes


class MediaPipePersonDetector(PersonDetector):
    """MediaPipe's full-body SSD person detector. Its own preprocessing
    handles letterboxing and its own postprocessing already maps boxes back
    to original-frame pixel coordinates (unlike NanoDet, no separate
    unletterbox step is needed here) -- see
    vendor/opencv_zoo/models/person_detection_mediapipe/mp_persondet.py.

    Output rows are [x1, y1, x2, y2, <8 landmark coords, unused here>, score].

    WARNING -- confirmed by live testing, not just reading the source: the
    (x1,y1,x2,y2) box is NOT a full-body box. It's a small SSD anchor-scale
    detection box centered near the face/chest; MediaPipe's own pipeline
    derives the actual full-body ROI from the 4 auxiliary landmarks this
    model also outputs (rows are [box(4), landmarks(8), score] -- see the
    vendored mp_persondet.py's comment: "each landmark: hip center point;
    full body point; shoulder center point; upper body point"). This class
    uses only the raw box, so anything built on it will crop roughly
    chest-and-above, not head-to-feet. NanoDetPersonDetector (COCO, true
    object-extent boxes) is the default for exactly this reason -- see
    docs/models.md. Fixing this properly would mean implementing MediaPipe's
    landmark-to-ROI conversion (center/size/rotation from the landmark
    pairs), which isn't in the vendored file and wasn't safe to guess at
    without a way to verify it against the model's actual training
    convention (see vendor_loader.py's docstring for why guessing at model
    internals is avoided here). Left in place as a fast alternative for
    anyone who wants to do that work.
    """

    def __init__(
        self,
        model_path: str | None = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ):
        mp_mod = load_module("person_detection_mediapipe/mp_persondet.py")
        # int8bq fails to load under this OpenCV build's ONNX importer
        # (DequantizeLinear parse error -- see docs/models.md); float32
        # measured fast enough on Pi 5 that this isn't a real loss.
        weights = model_path or model_weights_path(
            "person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx"
        )
        self._model = mp_mod.MPPersonDet(
            modelPath=weights, scoreThreshold=score_threshold, nmsThreshold=nms_threshold
        )

    def detect(self, rgb: np.ndarray) -> list[Box]:
        # MPPersonDet's own preprocessing does BGR->RGB internally, so it
        # expects a BGR frame the way the rest of opencv_zoo's demos feed it.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        preds = self._model.infer(bgr)
        boxes: list[Box] = []
        for pred in preds:
            x1, y1, x2, y2 = pred[:4]
            score = float(pred[-1])
            boxes.append(Box(int(x1), int(y1), int(x2 - x1), int(y2 - y1), score))
        return boxes


def build_detector(kind: str, **kwargs) -> PersonDetector:
    if kind == "nanodet":
        return NanoDetPersonDetector(**kwargs)
    if kind == "mediapipe":
        return MediaPipePersonDetector(**kwargs)
    raise ValueError(f"unknown detector kind: {kind!r}")
