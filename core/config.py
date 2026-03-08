"""
Configuration management system adapted for the new pipeline architecture.
Supports nested config.yaml schema with scenario overrides.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
import os
import logging
from threading import Lock


@dataclass
class Config:
    """Configuration settings for the assistive vision system."""

    # Detector settings
    detector_model_path: str = "yolov10n.pt"
    detector_confidence_threshold: float = 0.4

    # Depth estimation settings
    depth_model_name: str = "MiDaS_small"
    depth_frame_interval: int = 3

    # Tracker settings
    tracker_iou_threshold: float = 0.4
    tracker_cooldown_frames: int = 60

    # Speech settings
    speech_rate: int = 150
    speech_volume: float = 1.0

    # Camera settings
    camera_device_id: int = 0
    camera_width: int = 640
    camera_height: int = 480

    # Target classes
    target_classes: List[str] = field(default_factory=lambda: [
        "person", "chair", "table", "car", "bus", "bicycle",
        "motorcycle", "truck", "dog", "cat", "stairs", "door",
        "traffic light", "pole",
    ])

    # --- Legacy aliases kept so FrameProcessor still works ---
    @property
    def camera_index(self) -> int:
        return self.camera_device_id

    @property
    def frame_width(self) -> int:
        return self.camera_width

    @property
    def frame_height(self) -> int:
        return self.camera_height

    @property
    def model_name(self) -> str:
        return self.detector_model_path

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def confidence_threshold(self) -> float:
        return self.detector_confidence_threshold

    @property
    def voice_rate(self) -> int:
        return self.speech_rate

    @property
    def voice_volume(self) -> float:
        return self.speech_volume

    @property
    def announcement_cooldown(self) -> float:
        return float(self.tracker_cooldown_frames)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigManager:
    """Manage system configuration with YAML loading, scenario support, and validation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self._config_lock = Lock()
        self._raw: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_config(self, config_path: str | None = None, scenario: str = "default") -> Config:
        """Load configuration, applying scenario overrides on top of defaults."""
        with self._config_lock:
            if config_path:
                self.config_path = config_path

            if not os.path.exists(self.config_path):
                self.logger.info(f"Config file {self.config_path} not found, using defaults")
                self._apply_env_overrides()
                return self.config

            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._raw = yaml.safe_load(f) or {}

                # Start from defaults defined in the YAML "default" section
                merged: Dict[str, Any] = {}
                if "default" in self._raw and isinstance(self._raw["default"], dict):
                    merged = self._deep_merge(merged, self._raw["default"])

                # Apply scenario overrides (if not "default")
                if scenario != "default" and scenario in self._raw and isinstance(self._raw[scenario], dict):
                    merged = self._deep_merge(merged, self._raw[scenario])

                self._apply_merged(merged)
                self._apply_env_overrides()

                if not self.validate_config():
                    raise ConfigValidationError("Configuration validation failed")

                self.logger.info(f"Configuration loaded (scenario={scenario}) from {self.config_path}")
            except ConfigValidationError:
                raise
            except Exception as e:
                self.logger.error(f"Error loading config: {e}. Using defaults.")
                self.config = Config()

            return self.config

    def get_raw(self) -> Dict[str, Any]:
        """Return the raw parsed YAML dict (useful for VisionPipeline)."""
        return self._raw

    def validate_config(self) -> bool:
        """Validate current configuration values."""
        errors: list[str] = []

        if not isinstance(self.config.detector_model_path, str) or not self.config.detector_model_path.strip():
            errors.append("detector_model_path must be a non-empty string")
        if not 0.0 <= self.config.detector_confidence_threshold <= 1.0:
            errors.append("detector_confidence_threshold must be between 0.0 and 1.0")
        if self.config.depth_frame_interval < 1:
            errors.append("depth_frame_interval must be >= 1")
        if self.config.tracker_iou_threshold < 0.0 or self.config.tracker_iou_threshold > 1.0:
            errors.append("tracker_iou_threshold must be between 0.0 and 1.0")
        if self.config.tracker_cooldown_frames < 0:
            errors.append("tracker_cooldown_frames must be >= 0")
        if not 50 <= self.config.speech_rate <= 400:
            errors.append("speech_rate must be between 50 and 400")
        if not 0.0 <= self.config.speech_volume <= 1.0:
            errors.append("speech_volume must be between 0.0 and 1.0")
        if self.config.camera_device_id < 0:
            errors.append("camera_device_id must be >= 0")
        if self.config.camera_width < 32 or self.config.camera_height < 32:
            errors.append("camera dimensions must be >= 32")
        if not isinstance(self.config.target_classes, list) or len(self.config.target_classes) == 0:
            errors.append("target_classes must be a non-empty list")

        for err in errors:
            self.logger.error(f"Config validation: {err}")

        return len(errors) == 0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge *override* into *base*, returning a new dict."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_merged(self, d: Dict[str, Any]) -> None:
        """Map a merged nested dict onto the Config dataclass."""
        det = d.get("detector", {})
        if "model_path" in det:
            self.config.detector_model_path = det["model_path"]
        if "confidence_threshold" in det:
            self.config.detector_confidence_threshold = float(det["confidence_threshold"])

        depth = d.get("depth", {})
        if "model_name" in depth:
            self.config.depth_model_name = depth["model_name"]
        if "frame_interval" in depth:
            self.config.depth_frame_interval = int(depth["frame_interval"])

        tracker = d.get("tracker", {})
        if "iou_threshold" in tracker:
            self.config.tracker_iou_threshold = float(tracker["iou_threshold"])
        if "announcement_cooldown_frames" in tracker:
            self.config.tracker_cooldown_frames = int(tracker["announcement_cooldown_frames"])

        speech = d.get("speech", {})
        if "rate" in speech:
            self.config.speech_rate = int(speech["rate"])
        if "volume" in speech:
            self.config.speech_volume = float(speech["volume"])

        cam = d.get("camera", {})
        if "device_id" in cam:
            self.config.camera_device_id = int(cam["device_id"])
        if "width" in cam:
            self.config.camera_width = int(cam["width"])
        if "height" in cam:
            self.config.camera_height = int(cam["height"])

        if "target_classes" in d and isinstance(d["target_classes"], list):
            self.config.target_classes = d["target_classes"]

    def _apply_env_overrides(self) -> None:
        """Allow environment variables to override config values."""
        env_map = {
            "BLIND_CAP_MODEL_PATH": ("detector_model_path", str),
            "BLIND_CAP_CONFIDENCE": ("detector_confidence_threshold", float),
            "BLIND_CAP_CAMERA_ID": ("camera_device_id", int),
            "BLIND_CAP_SPEECH_RATE": ("speech_rate", int),
        }
        for env_key, (attr, converter) in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    setattr(self.config, attr, converter(val))
                    self.logger.info(f"Env override: {env_key}={val}")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid env override {env_key}={val}: {e}")
