"""
IoU-based object tracker with announcement gating.

If the `supervision` package is available, ByteTrack is used as the matching
backend.  Otherwise a simple greedy IoU matcher is used.
"""

from __future__ import annotations

import numpy as np
from core.logging_config import get_logger

logger = get_logger(__name__)

try:
    import supervision as sv
    _USE_BYTETRACK = True
    logger.info("supervision found — using ByteTrack backend")
except ImportError:
    _USE_BYTETRACK = False
    logger.info("supervision not found — using built-in IoU tracker")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two xyxy bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ObjectTracker:
    """Tracks detected objects across frames with ID assignment."""

    def __init__(self, iou_threshold: float = 0.4, cooldown_frames: int = 60) -> None:
        self._iou_threshold = iou_threshold
        self._cooldown_frames = cooldown_frames

        self._tracks: dict[int, dict] = {}
        self._next_id: int = 1
        self._announced: dict[int, int] = {}      # track_id → frame last announced
        self._frame_count: int = 0
        self._last_seen: dict[int, int] = {}       # track_id → frame last matched
        self._stale_limit: int = 30                 # prune after this many unmatched frames

        if _USE_BYTETRACK:
            self._byte_tracker = sv.ByteTrack()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, detections: list[dict]) -> list[dict]:
        """Match new detections to existing tracks and assign IDs.

        Each returned dict has an added ``track_id`` (int) key.
        """
        self._frame_count += 1

        if _USE_BYTETRACK:
            return self._update_bytetrack(detections)
        return self._update_iou(detections)

    def is_new(self, track_id: int) -> bool:
        """True if *track_id* has never been announced or is past cooldown."""
        if track_id not in self._announced:
            return True
        return (self._frame_count - self._announced[track_id]) >= self._cooldown_frames

    def mark_announced(self, track_id: int) -> None:
        """Record that this track was just announced."""
        self._announced[track_id] = self._frame_count

    # ------------------------------------------------------------------ #
    # ByteTrack backend
    # ------------------------------------------------------------------ #

    def _update_bytetrack(self, detections: list[dict]) -> list[dict]:
        if not detections:
            self._prune_stale()
            return []

        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confs = np.array([d["confidence"] for d in detections], dtype=np.float32)
        class_ids = np.array([d["class_id"] for d in detections], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=class_ids,
        )
        tracked = self._byte_tracker.update_with_detections(sv_dets)

        result: list[dict] = []
        for i in range(len(tracked)):
            det = dict(detections[i])
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else self._next_id
            det["track_id"] = tid
            self._tracks[tid] = det
            self._last_seen[tid] = self._frame_count
            result.append(det)

        self._prune_stale()
        return result

    # ------------------------------------------------------------------ #
    # Manual greedy IoU backend
    # ------------------------------------------------------------------ #

    def _update_iou(self, detections: list[dict]) -> list[dict]:
        if not detections:
            self._prune_stale()
            return []

        # Build cost matrix (negative IoU for greedy assignment)
        track_ids = list(self._tracks.keys())
        matched_det: set[int] = set()
        matched_trk: set[int] = set()
        assignments: list[tuple[int, int]] = []  # (det_idx, track_id)

        if track_ids:
            pairs: list[tuple[float, int, int]] = []
            for di, det in enumerate(detections):
                for tid in track_ids:
                    score = _iou(det["bbox"], self._tracks[tid]["bbox"])
                    if score >= self._iou_threshold:
                        pairs.append((score, di, tid))

            # Greedy: highest IoU first
            pairs.sort(key=lambda x: x[0], reverse=True)
            for score, di, tid in pairs:
                if di in matched_det or tid in matched_trk:
                    continue
                matched_det.add(di)
                matched_trk.add(tid)
                assignments.append((di, tid))

        # Build result
        result: list[dict] = []
        for di, det in enumerate(detections):
            d = dict(det)
            tid: int | None = None
            for a_di, a_tid in assignments:
                if a_di == di:
                    tid = a_tid
                    break
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            d["track_id"] = tid
            self._tracks[tid] = d
            self._last_seen[tid] = self._frame_count
            result.append(d)

        self._prune_stale()
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _prune_stale(self) -> None:
        """Remove tracks not seen for ``_stale_limit`` frames."""
        stale = [
            tid for tid, last in self._last_seen.items()
            if (self._frame_count - last) > self._stale_limit
        ]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._announced.pop(tid, None)
