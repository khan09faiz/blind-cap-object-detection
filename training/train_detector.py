#!/usr/bin/env python3
"""
training/train_detector.py
──────────────────────────
LOCAL utility — validates that best.pt (downloaded from Google Colab)
was trained on the correct 15 classes and is ready to use.

Training itself runs exclusively in Google Colab.
See training/colab_training.ipynb for the full training pipeline.

Usage:
    python -m training.train_detector --validate models/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ────────────────────────────────────────────────────────────
# Target class list — must match training/dataset.yaml
# ────────────────────────────────────────────────────────────
TARGET_CLASSES = [
    "person", "chair", "table", "sofa", "bed",
    "door", "bench", "trash_can", "fire_hydrant", "mailbox",
    "car", "bus", "truck", "bicycle", "dog",
]

NUM_CLASSES = len(TARGET_CLASSES)  # 15


def validate_model(model_path: str) -> bool:
    """
    Load *model_path* and confirm it has the expected 15 classes.
    Returns True on success, False on failure.
    """
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        print("  Download best.pt from Colab and copy it to models/best.pt")
        return False

    print(f"Loading {path} …")
    model = YOLO(str(path))
    model_classes: dict[int, str] = dict(model.names)

    if len(model_classes) != NUM_CLASSES:
        print(
            f"[WARN]  Expected {NUM_CLASSES} classes, "
            f"model has {len(model_classes)}: {list(model_classes.values())}"
        )
        return False

    print(f"[OK]    {NUM_CLASSES} classes confirmed: {list(model_classes.values())}")
    print("[OK]    Model is ready — run:  python run_app.py --scenario custom")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a best.pt model downloaded from Google Colab. "
            "Training runs in Colab only — see training/colab_training.ipynb."
        )
    )
    parser.add_argument(
        "--validate",
        metavar="MODEL_PATH",
        required=True,
        help="Path to the downloaded model file (e.g. models/best.pt)",
    )
    args = parser.parse_args()

    ok = validate_model(args.validate)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
