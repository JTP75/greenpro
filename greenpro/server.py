"""Local HTTP preview + live-config server.

Routes:
  /            HTML page showing raw / mask / grid side by side + stats
  /raw.mjpg    oriented + fitted camera preview, with the crop rect outlined
  /mask.mjpg   the 2-color segmentation mask at lores resolution
  /grid.mjpg   the 17x9 display payload, nearest-neighbor upscaled with gridlines
  /frame.json  {"w": 9, "h": 17, "rgb": [[r,g,b], ...]} -- 153 entries,
               row-major (row 0 first), the contract the future sender reads
  /config      GET current config as JSON; POST a partial JSON patch to apply live

This module intentionally has no dependency on any particular delivery
mechanism to the building -- see pipeline.Pipeline.latest_grid() for the
hand-off point a future sender will poll or subscribe to.
"""

from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .config import DISPLAY_COLS, DISPLAY_ROWS, LiveConfig
from .pipeline import LatestFrame, Pipeline

log = logging.getLogger(__name__)

BOUNDARY = "greenproframe"
GRID_UPSCALE = 30  # each grid cell rendered this many pixels wide in /grid.mjpg


def _encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _draw_crop_rect(rgb: np.ndarray, rect: tuple[int, int, int, int] | None) -> np.ndarray:
    if rect is None:
        return rgb
    out = rgb.copy()
    x, y, w, h = rect
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)
    return out


def _grid_preview(grid_rgb: np.ndarray) -> np.ndarray:
    up = cv2.resize(
        grid_rgb,
        (DISPLAY_COLS * GRID_UPSCALE, DISPLAY_ROWS * GRID_UPSCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    for i in range(DISPLAY_ROWS + 1):
        y = min(i * GRID_UPSCALE, up.shape[0] - 1)
        cv2.line(up, (0, y), (up.shape[1] - 1, y), (60, 60, 60), 1)
    for j in range(DISPLAY_COLS + 1):
        x = min(j * GRID_UPSCALE, up.shape[1] - 1)
        cv2.line(up, (x, 0), (x, up.shape[0] - 1), (60, 60, 60), 1)
    return up


def _frame_to_dict(frame: LatestFrame) -> dict:
    grid = frame.grid_rgb
    if grid is None:
        return {"w": DISPLAY_COLS, "h": DISPLAY_ROWS, "rgb": [], "frame_no": 0}
    flat = grid.reshape(-1, 3).tolist()
    return {
        "w": DISPLAY_COLS,
        "h": DISPLAY_ROWS,
        "rgb": flat,
        "frame_no": frame.frame_no,
        "fps": round(frame.timings.fps, 2),
        "inference_fps": round(frame.timings.inference_fps, 2),
    }


INDEX_HTML = """<!doctype html>
<html><head><title>greenpro preview</title>
<style>
  body {{ background:#111; color:#eee; font-family: system-ui, sans-serif; }}
  .row {{ display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }}
  img {{ image-rendering: pixelated; border:1px solid #444; }}
  .stats {{ font-family: monospace; white-space:pre; margin-top:12px; }}
  h3 {{ margin: 4px 0; }}
</style></head>
<body>
  <h2>greenpro pipeline preview</h2>
  <div class="row">
    <div><h3>raw</h3><img src="/raw.mjpg" width="{main_w}"></div>
    <div><h3>mask</h3><img src="/mask.mjpg" width="{lores_w}"></div>
    <div><h3>17x9 grid</h3><img src="/grid.mjpg"></div>
  </div>
  <div class="stats" id="stats">loading...</div>
  <script>
    async function poll() {{
      try {{
        const r = await fetch('/frame.json');
        const d = await r.json();
        document.getElementById('stats').textContent =
          `frame ${{d.frame_no}}  display_fps=${{d.fps}}  inference_fps=${{d.inference_fps}}`;
      }} catch (e) {{}}
      setTimeout(poll, 1000);
    }}
    poll();
  </script>
</body></html>
"""


def make_handler(pipeline: Pipeline, live_config: LiveConfig):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _stream_mjpeg(self, view: str) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            )
            self.end_headers()
            last_frame_no = -1
            cfg = live_config.get()
            try:
                while True:
                    frame = pipeline.holder.wait_next(last_frame_no, timeout=2.0)
                    if frame.frame_no == last_frame_no or frame.grid_rgb is None:
                        continue
                    last_frame_no = frame.frame_no

                    if view == "raw" and frame.raw_rgb is not None:
                        img = _draw_crop_rect(frame.raw_rgb, frame.crop_rect)
                    elif view == "mask" and frame.mask_rgb is not None:
                        img = frame.mask_rgb
                    elif view == "grid" and frame.grid_rgb is not None:
                        img = _grid_preview(frame.grid_rgb)
                    else:
                        continue

                    jpeg = _encode_jpeg(img, cfg.server.jpeg_quality)
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except ConnectionError:
                # Client disconnected mid-stream (closed tab, network drop,
                # etc). Covers BrokenPipeError/ConnectionResetError/
                # ConnectionAbortedError -- all normal, not worth logging.
                pass

        def do_GET(self):
            if self.path == "/":
                cfg = live_config.get()
                body = INDEX_HTML.format(
                    main_w=cfg.capture.main_size[0], lores_w=cfg.capture.lores_size[0]
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path in ("/raw.mjpg", "/mask.mjpg", "/grid.mjpg"):
                self._stream_mjpeg(self.path.split("/")[1].split(".")[0])
            elif self.path == "/frame.json":
                self._send_json(200, _frame_to_dict(pipeline.holder.get()))
            elif self.path == "/config":
                self._send_json(200, live_config.get().to_dict())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/config":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                patch = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                self._send_json(400, {"error": f"invalid JSON: {exc}"})
                return
            new_cfg = live_config.apply_patch(patch)
            self._send_json(200, new_cfg.to_dict())

    return Handler


def run_server(pipeline: Pipeline, live_config: LiveConfig) -> None:
    cfg = live_config.get()
    handler = make_handler(pipeline, live_config)
    httpd = ThreadingHTTPServer((cfg.server.host, cfg.server.port), handler)
    log.info("serving on http://%s:%d/", cfg.server.host, cfg.server.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
