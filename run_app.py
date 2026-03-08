#!/usr/bin/env python3
"""
run_app.py — Convenience launcher for the Real-Time Assistive Vision System.
Delegates to main.run().
"""

import sys
import os
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-Time Assistive Vision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_app.py                          # default scenario
  python run_app.py --scenario indoor        # indoor scenario
  python run_app.py --config my_config.yaml  # custom config file
""",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--scenario",
        default="default",
        choices=["default", "indoor", "outdoor", "high_performance", "custom"],
        help="Configuration scenario (default: default)",
    )
    args = parser.parse_args()

    try:
        from main import run
        run(config_path=args.config, scenario=args.scenario)
    except KeyboardInterrupt:
        print("\nShutting down…")
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
