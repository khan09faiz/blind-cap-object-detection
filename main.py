#!/usr/bin/env python3
"""
main.py — Webcam entry point for the Real-Time Assistive Vision System.

Loads configuration, creates ``VisionPipeline``, captures frames, draws
simple bounding-box overlays, and exits on ESC key.  All detection, depth,
tracking, and speech logic lives inside the pipeline — not here.
"""

from __future__ import annotations

import signal
import sys

import cv2
import numpy as np
import yaml

from core.config import ConfigManager
from core.logging_config import initialize_logging, get_logger, shutdown_logging
from pipeline.vision_pipeline import VisionPipeline

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Globals for signal handling
# ------------------------------------------------------------------ #
_running = True


def _signal_handler(sig: int, frame: object) -> None:
    global _running
    _running = False


# ------------------------------------------------------------------ #
# Drawing helper
# ------------------------------------------------------------------ #

def _draw_results(
    frame: np.ndarray,
    tracked: list[dict],
    results: list[tuple[str, str, float]],
) -> None:
    """Draw bounding boxes and labels on *frame* in-place."""
    for det, (label, direction, distance) in zip(tracked, results):
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        # Colour by distance
        if distance < 2.0:
            colour = (0, 0, 255)    # red — close
        elif distance < 5.0:
            colour = (0, 165, 255)  # orange — medium
        else:
            colour = (0, 255, 0)    # green — far

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        text = f"{label} {direction} {distance:.1f}m"
        cv2.putText(
            frame, text, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
        )


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def run(config_path: str = "config.yaml", scenario: str = "default") -> None:
    """Run the assistive vision system."""
    global _running

    # Logging
    initialize_logging()
    logger.info("Starting Assistive Vision System")

    # Config
    cm = ConfigManager(config_path)
    cfg = cm.load_config(scenario=scenario)
    raw = cm.get_raw()

    # Merge scenario on top of default for pipeline dict
    from core.config import ConfigManager as _CM
    pipeline_cfg = _CM._deep_merge(
        raw.get("default", {}),
        raw.get(scenario, {}) if scenario != "default" else {},
    )

    # Pipeline
    pipeline = VisionPipeline(pipeline_cfg)

    # Camera
    cap = cv2.VideoCapture(cfg.camera_device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)

    if not cap.isOpened():
        logger.critical("Cannot open camera %d", cfg.camera_device_id)
        sys.exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("Camera opened — entering main loop (ESC to quit)")

    try:
        while _running:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Frame capture failed, retrying…")
                continue

            results = pipeline.process(frame)

            # stdout output
            for label, direction, distance in results:
                print(f"{label} {direction} at {distance:.1f} meters")

            # We need tracked dicts for drawing. Re-run tracker won't help,
            # so we store a lightweight copy inside the pipeline.
            # Alternatively, just draw from results tuple:
            _draw_results_simple(frame, results)

            cv2.imshow("Blind Cap", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        pipeline.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        shutdown_logging()
        logger.info("System exited cleanly")


def _draw_results_simple(
    frame: np.ndarray,
    results: list[tuple[str, str, float]],
) -> None:
    """Minimal draw when we only have (label, direction, distance)."""
    y = 30
    for label, direction, distance in results:
        if distance < 2.0:
            colour = (0, 0, 255)
        elif distance < 5.0:
            colour = (0, 165, 255)
        else:
            colour = (0, 255, 0)

        text = f"{label} {direction} {distance:.1f}m"
        cv2.putText(
            frame, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
        )
        y += 28


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-Time Assistive Vision System")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument(
        "--scenario",
        default="default",
        choices=["default", "indoor", "outdoor", "high_performance", "custom"],
        help="Configuration scenario",
    )
    args = parser.parse_args()
    run(config_path=args.config, scenario=args.scenario)
