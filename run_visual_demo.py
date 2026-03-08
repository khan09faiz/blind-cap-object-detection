#!/usr/bin/env python3
"""
run_visual_demo.py — Visual demo for the Real-Time Assistive Vision System.
Same as run_app.py (the main pipeline already shows a cv2 window).
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner() -> None:
    print("=" * 60)
    print("  ASSISTIVE VISION SYSTEM — VISUAL DEMO")
    print("=" * 60)
    print("  Controls:  ESC — quit")
    print("=" * 60)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visual demo — Real-Time Assistive Vision System",
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument(
        "--scenario",
        default="default",
        choices=["default", "indoor", "outdoor", "high_performance", "custom"],
        help="Configuration scenario",
    )
    args = parser.parse_args()

    print_banner()

    try:
        from main import run
        run(config_path=args.config, scenario=args.scenario)
    except KeyboardInterrupt:
        print("\nDemo stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
