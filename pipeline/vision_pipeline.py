"""
VisionPipeline — the single orchestrator that wires detection, depth estimation,
tracking, direction classification, and speech together.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.detector import ObjectDetector
from models.depth_estimator import DepthEstimator, estimate_object_distance
from tracking.tracker import ObjectTracker
from audio.speech import SpeechEngine, create_message
from utils.direction import get_direction
from utils.obstacle_rules import priority_score
from core.logging_config import get_logger

logger = get_logger(__name__)

# Optional performance integration
try:
    from core.performance import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None  # type: ignore[assignment,misc]

# Optional navigation integration
try:
    from core.navigation import NavigationGuidanceSystem
except ImportError:
    NavigationGuidanceSystem = None  # type: ignore[assignment,misc]


class VisionPipeline:
    """Real-time assistive vision pipeline.

    Instantiate once at startup; call ``process(frame)`` on each camera frame.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        det_cfg = config.get("detector", {})
        depth_cfg = config.get("depth", {})
        trk_cfg = config.get("tracker", {})
        spk_cfg = config.get("speech", {})

        # Optional class filter — only announce objects in this set.
        # Empty set means "announce everything the model detects".
        raw_classes: list = config.get("target_classes", [])
        self._target_classes: set[str] = {c.lower() for c in raw_classes} if raw_classes else set()

        # Sub-components (all I/O happens here, not per-frame)
        self.detector = ObjectDetector(
            model_path=det_cfg.get("model_path", "yolov10n.pt"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.4),
        )
        self.depth_estimator = DepthEstimator(
            model_name=depth_cfg.get("model_name", "MiDaS_small"),
        )
        self.tracker = ObjectTracker(
            iou_threshold=trk_cfg.get("iou_threshold", 0.4),
            cooldown_frames=trk_cfg.get("announcement_cooldown_frames", 60),
        )
        self.speech = SpeechEngine(
            rate=spk_cfg.get("rate", 150),
            volume=spk_cfg.get("volume", 1.0),
        )

        # Depth throttling
        self._depth_interval: int = depth_cfg.get("frame_interval", 3)
        self._frame_count: int = 0
        self._last_depth_map: np.ndarray | None = None

        # Optional performance monitor
        self._perf: Any = None
        if PerformanceMonitor is not None:
            try:
                self._perf = PerformanceMonitor()
            except Exception:
                pass

        # Optional navigation guidance
        self._nav: Any = None
        if NavigationGuidanceSystem is not None:
            try:
                self._nav = NavigationGuidanceSystem()
            except Exception:
                pass

        logger.info("VisionPipeline initialised")

    # ------------------------------------------------------------------ #
    # Per-frame processing
    # ------------------------------------------------------------------ #

    def process(self, frame: np.ndarray) -> list[tuple[str, str, float]]:
        """Run the full detect → track → depth → speak pipeline on *frame*.

        Returns a list of ``(label, direction, distance)`` sorted by priority
        (dangerous objects closest to the camera are spoken first).
        """
        self._frame_count += 1
        ctx = self._perf.measure_frame() if self._perf else _noop_ctx()

        with ctx:
            # 1. Detection
            detections = self.detector.detect(frame)

            # 2. Tracking
            tracked = self.tracker.update(detections)

            # 3. Depth (every N frames)
            if self._frame_count % self._depth_interval == 0 or self._last_depth_map is None:
                try:
                    self._last_depth_map = self.depth_estimator.estimate(frame)
                except Exception as e:
                    logger.warning(f"Depth estimation failed: {e}")

            # 4. Collect all objects with their properties
            candidates: list[dict] = []
            frame_w = frame.shape[1]

            for det in tracked:
                bbox  = det["bbox"]
                label = det["label"]

                if self._target_classes and label.lower() not in self._target_classes:
                    continue

                direction = get_direction(bbox, frame_w)
                distance  = (
                    estimate_object_distance(self._last_depth_map, bbox)
                    if self._last_depth_map is not None
                    else 0.0
                )

                candidates.append({
                    "label":     label,
                    "direction": direction,
                    "distance":  distance,
                    "track_id":  det["track_id"],
                    "priority":  priority_score(label, distance),
                })

            # 5. Sort: lowest priority_score first
            #    → Tier-1 fast vehicles first, then Tier-2 obstacles,
            #      then safe objects; within each tier, closer comes first.
            candidates.sort(key=lambda c: c["priority"])

            # 6. Speak in priority order (only new / cooldown-expired objects)
            for c in candidates:
                msg = create_message(c["label"], c["direction"], c["distance"])
                if self.tracker.is_new(c["track_id"]):
                    self.speech.speak(msg)
                    self.tracker.mark_announced(c["track_id"])

            results = [(c["label"], c["direction"], c["distance"]) for c in candidates]

        return results

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:
        """Clean up resources (speech thread, etc.)."""
        self.speech.shutdown()
        logger.info("VisionPipeline shut down")


# ------------------------------------------------------------------ #
# Tiny helper so the timing context-manager is always available
# ------------------------------------------------------------------ #

class _noop_ctx:
    """No-op context manager used when PerformanceMonitor is absent."""
    def __enter__(self) -> None:
        return None
    def __exit__(self, *_: object) -> None:
        pass
