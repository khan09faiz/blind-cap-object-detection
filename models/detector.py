"""
YOLOv10 object detection wrapper.
All inference runs on CPU.

Supports two weight sources:
  • yolov10n.pt / yolov10s.pt  — COCO-pretrained baseline (80 classes)
  • models/best.pt             — Custom-trained on Open Images (15 classes)

The detector auto-detects which weight file is loaded by examining model.names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from core.logging_config import get_logger

logger = get_logger(__name__)

# Expected number of classes in the custom-trained best.pt.
# Used only to decide whether to log a "custom model" message.
_CUSTOM_CLASS_COUNT = 15


class ObjectDetector:
    """YOLOv10-based object detector running exclusively on CPU.

    Automatically uses CUSTOM_CLASS_NAMES when a 15-class model (best.pt)
    is loaded, and falls back to the model's built-in names otherwise.
    """

    def __init__(self, model_path: str = "yolov10n.pt", confidence_threshold: float = 0.4) -> None:
        self.confidence_threshold = confidence_threshold
        logger.info(f"Loading YOLO model: {model_path} (device=cpu)")
        self.model = YOLO(model_path)
        self.model.to("cpu")

        # Always trust the class names baked into the weight file.
        self._class_names: dict[int, str] = dict(self.model.names)
        if len(self._class_names) == _CUSTOM_CLASS_COUNT and Path(model_path).stem in ("best", "last"):
            logger.info("Detected custom 15-class model")
        logger.info(f"Model loaded — {len(self._class_names)} classes: {list(self._class_names.values())}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on a BGR frame.

        Returns a list of dicts with keys:
            bbox (np.ndarray): xyxy, shape (4,)
            class_id (int)
            label (str)
            confidence (float)
        """
        results = self.model(frame, verbose=False, device="cpu")
        detections: list[dict] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0].cpu().item())
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls[0].cpu().item())
                detections.append({
                    "bbox": box.xyxy[0].cpu().numpy().astype(np.float32),
                    "class_id": cls_id,
                    "label": self._class_names.get(cls_id, f"class_{cls_id}"),
                    "confidence": conf,
                })

        return detections

    def get_class_names(self) -> dict[int, str]:
        """Return the model's class-name mapping."""
        return dict(self._class_names)
