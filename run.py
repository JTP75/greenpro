#!/usr/bin/env python3
"""Entry point for the greenpro capture/processing pipeline.

Usage:
    python run.py                                # use config.yaml, real camera
    python run.py --source image:picam_snap.jpg  # no camera needed
    python run.py --source video:clip.mp4
    python run.py --config myconfig.yaml --port 8080

Then open http://<host>:8000/ (or the --port you chose) to see the raw feed,
the 2-color mask, and the 17x9 grid update live, and to read /frame.json.
"""

from __future__ import annotations

import argparse
import logging
import sys

from greenpro import config as config_module
from greenpro.pipeline import Pipeline
from greenpro.server import run_server


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config.yaml", help="path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--source",
        default=None,
        help='override capture.source, e.g. "picam", "image:path.jpg", "video:clip.mp4"',
    )
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = config_module.load(args.config)
    if args.source is not None:
        cfg.capture.source = args.source
    if args.port is not None:
        cfg.server.port = args.port
    cfg.clamp()

    live_config = config_module.LiveConfig(cfg)
    pipeline = Pipeline(live_config)
    pipeline.start()
    try:
        run_server(pipeline, live_config)
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
