#!/usr/bin/env python3
"""Entry point for the greenpro capture/processing pipeline.

Usage:
    python run.py                                # use config.yaml, real camera
    python run.py --source image:picam_snap.jpg  # no camera needed
    python run.py --source video:clip.mp4
    python run.py --source pattern:blink          # synthetic grid, no camera/segmenter
    python run.py --config myconfig.yaml --port 8080
    python run.py --source pattern:corners --send --instance hazel-toad

Then open http://<host>:8000/ (or the --port you chose) to see the raw feed,
the 2-color mask, and the 17x9 grid update live, and to read /frame.json.

--send starts a background sender that POSTs each new grid to Will's Green
Building simulator (see docs/simulator.md). It runs regardless of --send --
sending only happens once sink.enabled is true -- so it can also be turned
on/off live, without a restart, via:
    curl -X POST http://localhost:8000/config -d '{"sink": {"enabled": true}}'
"""

from __future__ import annotations

import argparse
import logging
import sys

from greenpro import config as config_module
from greenpro.patterns import PatternPipeline, build_renderer
from greenpro.pipeline import Pipeline
from greenpro.server import run_server
from greenpro.sink import FrameSender


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", default="config.yaml", help="path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            'override capture.source, e.g. "picam", "image:path.jpg", '
            '"video:clip.mp4", "pattern:blink" (see greenpro/patterns.py '
            "for all pattern names)"
        ),
    )
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument(
        "--send",
        action="store_true",
        help="set sink.enabled=true at startup (frames POST to sink.instance immediately)",
    )
    parser.add_argument(
        "--instance", default=None, help="override sink.instance, e.g. hazel-toad"
    )
    parser.add_argument(
        "--sink-url",
        default=None,
        help=f"override sink.base_url (config.yaml default: {config_module.SinkConfig().base_url})",
    )
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
    if args.instance is not None:
        cfg.sink.instance = args.instance
    if args.sink_url is not None:
        cfg.sink.base_url = args.sink_url
    if args.send:
        cfg.sink.enabled = True
    cfg.clamp()

    live_config = config_module.LiveConfig(cfg)

    if cfg.capture.source.startswith("pattern:"):
        pipeline = PatternPipeline(live_config, build_renderer(cfg.capture.source.split(":", 1)[1]))
    else:
        pipeline = Pipeline(live_config)

    # Always started -- sending only happens once sink.enabled is true (see
    # FrameSender._run), so this is what makes `POST /config
    # {"sink":{"enabled":true}}` work live without a restart.
    sender = FrameSender(pipeline.holder, live_config)

    pipeline.start()
    sender.start()
    try:
        run_server(pipeline, live_config, sender)
    finally:
        sender.stop()
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
