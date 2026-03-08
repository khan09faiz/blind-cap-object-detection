"""
Offline non-blocking TTS engine using pyttsx3 + daemon thread + queue.
"""

from __future__ import annotations

import queue
import threading
from core.logging_config import get_logger
from utils.obstacle_rules import is_dangerous

logger = get_logger(__name__)

_SHUTDOWN_SENTINEL = object()


class SpeechEngine:
    """Non-blocking speech engine.  All pyttsx3 usage is confined to a single daemon thread."""

    def __init__(self, rate: int = 150, volume: float = 1.0) -> None:
        self._rate = rate
        self._volume = volume
        self._queue: queue.Queue = queue.Queue()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("SpeechEngine started (daemon thread)")

    def speak(self, text: str) -> None:
        """Queue *text* for speech.  Returns immediately."""
        self._queue.put(text)

    def shutdown(self) -> None:
        """Send sentinel to stop the worker thread gracefully."""
        self._queue.put(_SHUTDOWN_SENTINEL)
        self._thread.join(timeout=5)
        logger.info("SpeechEngine shut down")

    # ------------------------------------------------------------------ #
    # Worker (runs in daemon thread)
    # ------------------------------------------------------------------ #

    def _worker(self) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        while True:
            item = self._queue.get()
            if item is _SHUTDOWN_SENTINEL:
                break
            try:
                engine.say(item)
                engine.runAndWait()
            except Exception as exc:
                logger.error(f"TTS error: {exc}")


def create_message(
    label: str,
    direction: str,
    distance: float,
    dangerous: bool | None = None,
) -> str:
    """Build a spoken announcement string.

    Args:
        label: Object class name.
        direction: "left", "right", or "ahead".
        distance: Estimated distance in metres.
        dangerous: Override for danger flag.  When *None* (default),
            ``is_dangerous(label)`` is used.

    Returns:
        A human-readable message suitable for TTS.
    """
    if dangerous is None:
        dangerous = is_dangerous(label)

    if dangerous:
        return f"Warning — {label} {direction}"
    return f"{label} {direction} at {distance:.1f} meters"
