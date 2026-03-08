#!/usr/bin/env python3
"""
launch.py — Auto-detecting launcher for the Assistive Vision System.
Probes hardware (camera resolution) and picks a scenario accordingly.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def detect_scenario() -> str:
    """Return a reasonable scenario based on camera capability."""
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "default"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return "outdoor" if w >= 1280 else "default"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assistive Vision System — auto-detecting launcher",
    )
    parser.add_argument(
        "--scenario",
        choices=["auto", "default", "indoor", "outdoor", "high_performance", "custom"],
        default="auto",
        help="Scenario (default: auto-detect)",
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    args = parser.parse_args()

    scenario = args.scenario
    if scenario == "auto":
        scenario = detect_scenario()
        print(f"Auto-detected scenario: {scenario}")

    try:
        from main import run
        run(config_path=args.config, scenario=scenario)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
