"""Benchmark the candidate neural backends against a real frame, on real
hardware, before committing to any of them as the default. See docs/models.md
for the results this produces and Step 1 of the Phase 2 plan for why this
exists: every "Pi 5 estimate" in the design was extrapolated from Pi 4B
numbers and needed replacing with a measurement.

Usage:
    python -m greenpro.bench                       # uses the live camera
    python -m greenpro.bench --image picam_snap.jpg
    python -m greenpro.bench --iters 30
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2
import numpy as np

from .detectors import MediaPipePersonDetector, NanoDetPersonDetector
from .vendor_loader import load_module, model_weights_path


def _time_calls(fn, iters: int) -> dict:
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": times[min(len(times) - 1, int(len(times) * 0.95))],
        "min": times[0],
        "max": times[-1],
    }


def _fmt(stats: dict) -> str:
    fps = 1000.0 / stats["mean"] if stats["mean"] > 0 else float("inf")
    return (
        f"mean={stats['mean']:7.2f}ms  median={stats['median']:7.2f}ms  "
        f"p95={stats['p95']:7.2f}ms  ({fps:5.1f} fps)"
    )


def _get_frame(image_path: str | None) -> np.ndarray:
    """Returns an RGB frame, from --image or a single live camera capture."""
    if image_path:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    from picamera2 import Picamera2

    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (648, 486), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # let AE/AWB settle
    frame = picam2.capture_array("main")
    picam2.stop()
    picam2.close()
    return frame


def bench_pphumanseg(rgb: np.ndarray, weights: str, iters: int) -> dict:
    pphumanseg_mod = load_module("human_segmentation_pphumanseg/pphumanseg.py")
    model = pphumanseg_mod.PPHumanSeg(modelPath=weights)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Representative of TopDownSegmenter's real input: a person-sized crop,
    # not the full frame (PP-HumanSeg is a portrait model -- see detectors.py).
    h, w = bgr.shape[:2]
    crop = bgr[h // 4 : h, w // 3 : 2 * w // 3]
    return _time_calls(lambda: model.infer(crop), iters)


def bench_nanodet(rgb: np.ndarray, weights: str, iters: int) -> dict:
    detector = NanoDetPersonDetector(model_path=weights)
    return _time_calls(lambda: detector.detect(rgb), iters)


def bench_mediapipe_detector(rgb: np.ndarray, weights: str, iters: int) -> dict:
    detector = MediaPipePersonDetector(model_path=weights)
    return _time_calls(lambda: detector.detect(rgb), iters)


def bench_topdown_composed(rgb: np.ndarray, detector_kind: str, max_crops: int, iters: int) -> dict:
    """Detector + PP-HumanSeg on up to max_crops boxes -- the actual
    TopDownSegmenter workload, not just its parts in isolation."""
    if detector_kind == "nanodet":
        detector = NanoDetPersonDetector()
    else:
        detector = MediaPipePersonDetector()
    pphumanseg_mod = load_module("human_segmentation_pphumanseg/pphumanseg.py")
    # int8bq fails to load under cv2.dnn's ONNX importer on this OpenCV build
    # (DequantizeLinear parse error) -- see docs/models.md. float32 it is;
    # it's already fast enough that this isn't a real loss.
    seg_weights = model_weights_path(
        "human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx"
    )
    segmenter = pphumanseg_mod.PPHumanSeg(modelPath=seg_weights)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def run_once():
        boxes = sorted(detector.detect(rgb), key=lambda b: b.area(), reverse=True)[:max_crops]
        for box in boxes:
            b = box.clipped(bgr.shape[1], bgr.shape[0])
            if b.w > 0 and b.h > 0:
                segmenter.infer(bgr[b.y : b.y + b.h, b.x : b.x + b.w])
        if not boxes:
            # No one in frame during the benchmark capture -- still measure
            # detector-only cost so the number isn't silently zero.
            pass

    return _time_calls(run_once, iters)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="path to a still image instead of the live camera")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args(argv)

    print("Capturing frame..." if args.image is None else f"Loading {args.image}...")
    rgb = _get_frame(args.image)
    print(f"Frame shape: {rgb.shape}\n")

    results = {}

    print("== PP-HumanSeg (segmentation, on a person-sized crop) ==")
    for label, fname in [
        ("float32", "human_segmentation_pphumanseg_2023mar.onnx"),
        ("int8bq", "human_segmentation_pphumanseg_2023mar_int8bq.onnx"),
    ]:
        weights = model_weights_path(f"human_segmentation_pphumanseg/{fname}")
        try:
            stats = bench_pphumanseg(rgb, weights, args.iters)
            results[f"pphumanseg_{label}"] = stats
            print(f"  {label:8s} {_fmt(stats)}")
        except Exception as exc:
            print(f"  {label:8s} FAILED: {exc}")

    print("\n== NanoDet (person detection, full frame @ 416x416) ==")
    for label, fname in [
        ("float32", "object_detection_nanodet_2022nov.onnx"),
        ("int8bq", "object_detection_nanodet_2022nov_int8bq.onnx"),
    ]:
        weights = model_weights_path(f"object_detection_nanodet/{fname}")
        try:
            stats = bench_nanodet(rgb, weights, args.iters)
            results[f"nanodet_{label}"] = stats
            print(f"  {label:8s} {_fmt(stats)}")
        except Exception as exc:
            print(f"  {label:8s} FAILED: {exc}")

    print("\n== MediaPipe person detector (full frame @ 224x224) ==")
    for label, fname in [
        ("float32", "person_detection_mediapipe_2023mar.onnx"),
        ("int8bq", "person_detection_mediapipe_2023mar_int8bq.onnx"),
    ]:
        weights = model_weights_path(f"person_detection_mediapipe/{fname}")
        try:
            stats = bench_mediapipe_detector(rgb, weights, args.iters)
            results[f"mediapipe_{label}"] = stats
            print(f"  {label:8s} {_fmt(stats)}")
        except Exception as exc:
            print(f"  {label:8s} FAILED: {exc}")

    print("\n== Composed top-down pipeline (detect + segment top-K crops) ==")
    for detector_kind in ("nanodet", "mediapipe"):
        for max_crops in (1, 3):
            try:
                stats = bench_topdown_composed(rgb, detector_kind, max_crops, args.iters)
                results[f"topdown_{detector_kind}_k{max_crops}"] = stats
                print(f"  {detector_kind:9s} k={max_crops}  {_fmt(stats)}")
            except Exception as exc:
                print(f"  {detector_kind:9s} k={max_crops}  FAILED: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
