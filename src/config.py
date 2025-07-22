"""
Configuration management system with default settings.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
import yaml
import json
import os
from pathlib import Path
import logging
import time
from threading import Lock


@dataclass
class Config:
    """Configuration settings for the enhanced blind detection system."""
    
    # Model settings
    model_name: str = 'yolov8n.pt'
    confidence_threshold: float = 0.5
    device: str = 'auto'  # auto/cpu/cuda
    
    # Audio settings
    voice_rate: int = 200
    voice_volume: float = 0.9
    announcement_cooldown: float = 2.0
    
    # Detection settings
    target_classes: List[str] = field(default_factory=lambda: [
        'person', 'chair', 'table', 'sofa', 'bed', 'toilet', 'tv',
        'laptop', 'mouse', 'keyboard', 'cell phone', 'book', 'clock',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ])
    
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    
    # Spatial settings
    distance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'close': 0.3,  # bbox area > 30% of frame
        'medium': 0.1,  # bbox area 10-30% of frame
        'far': 0.0     # bbox area < 10% of frame
    })


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigManager:
    """Manage system configuration with file loading and validation."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self._config_lock = Lock()
        self._last_modified = 0
        self._watch_config = False
    
    def load_config(self, config_path: str = None) -> Config:
        """Load configuration from file, falling back to defaults."""
        with self._config_lock:
            if config_path:
                self.config_path = config_path
            
            if os.path.exists(self.config_path):
                try:
                    config_data = self._load_config_file(self.config_path)
                    self._merge_config_data(config_data)
                    
                    # Apply environment variable overrides
                    self._apply_env_overrides()
                    
                    # Validate loaded configuration
                    if not self.validate_config():
                        raise ConfigValidationError("Configuration validation failed")
                    
                    # Update last modified time for watching
                    self._last_modified = os.path.getmtime(self.config_path)
                    
                    self.logger.info(f"Configuration loaded from {self.config_path}")
                    print(f"Configuration loaded from {self.config_path}")
                except Exception as e:
                    self.logger.error(f"Error loading config file {self.config_path}: {e}")
                    print(f"Error loading config file {self.config_path}: {e}")
                    print("Using default configuration")
                    self.config = Config()  # Reset to defaults
            else:
                self.logger.info(f"Config file {self.config_path} not found, using defaults")
                print(f"Config file {self.config_path} not found, using defaults")
                
                # Still apply environment overrides even with default config
                self._apply_env_overrides()
            
            return self.config
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration data from YAML or JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f) or {}
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
    
    def _merge_config_data(self, config_data: Dict[str, Any]) -> None:
        """Merge loaded configuration data with default config."""
        for key, value in config_data.items():
            if hasattr(self.config, key):
                # Type validation for specific fields
                if key == 'target_classes' and not isinstance(value, list):
                    raise ConfigValidationError(f"target_classes must be a list, got {type(value)}")
                if key == 'distance_thresholds' and not isinstance(value, dict):
                    raise ConfigValidationError(f"distance_thresholds must be a dict, got {type(value)}")
                
                setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
                print(f"Warning: Unknown configuration key: {key}")
    
    def save_config(self, config_path: str = None) -> None:
        """Save current configuration to file."""
        if config_path:
            self.config_path = config_path
        
        try:
            config_dict = {
                'model_name': self.config.model_name,
                'confidence_threshold': self.config.confidence_threshold,
                'device': self.config.device,
                'voice_rate': self.config.voice_rate,
                'voice_volume': self.config.voice_volume,
                'announcement_cooldown': self.config.announcement_cooldown,
                'target_classes': self.config.target_classes,
                'camera_index': self.config.camera_index,
                'frame_width': self.config.frame_width,
                'frame_height': self.config.frame_height,
                'distance_thresholds': self.config.distance_thresholds
            }
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    json.dump(config_dict, f, indent=2)
            
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving config file {self.config_path}: {e}")
    
    def validate_config(self) -> bool:
        """Validate configuration settings with comprehensive checks."""
        validation_errors = []
        
        try:
            # Validate model settings
            if not isinstance(self.config.model_name, str) or not self.config.model_name.strip():
                validation_errors.append("model_name must be a non-empty string")
            
            if not isinstance(self.config.confidence_threshold, (int, float)):
                validation_errors.append("confidence_threshold must be a number")
            elif not 0.0 <= self.config.confidence_threshold <= 1.0:
                validation_errors.append("confidence_threshold must be between 0.0 and 1.0")
            
            if self.config.device not in ['auto', 'cpu', 'cuda']:
                validation_errors.append("device must be 'auto', 'cpu', or 'cuda'")
            
            # Validate audio settings
            if not isinstance(self.config.voice_rate, int):
                validation_errors.append("voice_rate must be an integer")
            elif not 50 <= self.config.voice_rate <= 400:
                validation_errors.append("voice_rate must be between 50 and 400")
            
            if not isinstance(self.config.voice_volume, (int, float)):
                validation_errors.append("voice_volume must be a number")
            elif not 0.0 <= self.config.voice_volume <= 1.0:
                validation_errors.append("voice_volume must be between 0.0 and 1.0")
            
            if not isinstance(self.config.announcement_cooldown, (int, float)):
                validation_errors.append("announcement_cooldown must be a number")
            elif self.config.announcement_cooldown < 0:
                validation_errors.append("announcement_cooldown must be non-negative")
            
            # Validate detection settings
            if not isinstance(self.config.target_classes, list):
                validation_errors.append("target_classes must be a list")
            elif not all(isinstance(cls, str) for cls in self.config.target_classes):
                validation_errors.append("all target_classes must be strings")
            elif len(self.config.target_classes) == 0:
                validation_errors.append("target_classes cannot be empty")
            
            # Validate camera settings
            if not isinstance(self.config.camera_index, int):
                validation_errors.append("camera_index must be an integer")
            elif self.config.camera_index < 0:
                validation_errors.append("camera_index must be non-negative")
            
            if not isinstance(self.config.frame_width, int) or not isinstance(self.config.frame_height, int):
                validation_errors.append("frame dimensions must be integers")
            elif self.config.frame_width <= 0 or self.config.frame_height <= 0:
                validation_errors.append("frame dimensions must be positive")
            elif self.config.frame_width < 320 or self.config.frame_height < 240:
                validation_errors.append("frame dimensions should be at least 320x240 for reliable detection")
            
            # Validate spatial settings
            if not isinstance(self.config.distance_thresholds, dict):
                validation_errors.append("distance_thresholds must be a dictionary")
            else:
                required_keys = {'close', 'medium', 'far'}
                if not required_keys.issubset(self.config.distance_thresholds.keys()):
                    validation_errors.append(f"distance_thresholds must contain keys: {required_keys}")
                
                for key, value in self.config.distance_thresholds.items():
                    if not isinstance(value, (int, float)):
                        validation_errors.append(f"distance_thresholds[{key}] must be a number")
                    elif value < 0 or value > 1:
                        validation_errors.append(f"distance_thresholds[{key}] must be between 0 and 1")
                
                # Validate threshold ordering
                if (self.config.distance_thresholds.get('close', 0) <= 
                    self.config.distance_thresholds.get('medium', 0)):
                    validation_errors.append("close threshold must be greater than medium threshold")
            
            if validation_errors:
                for error in validation_errors:
                    self.logger.error(f"Configuration validation error: {error}")
                    print(f"Configuration validation error: {error}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Unexpected error during validation: {e}")
            print(f"Unexpected error during validation: {e}")
            return False
    
    def get_config(self) -> Config:
        """Get current configuration."""
        return self.config
    
    def create_example_config(self, output_path: str = 'config_example.yaml') -> None:
        """Create an example configuration file with all options documented."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write comprehensive YAML with comments and environment variable info
                f.write("# Enhanced Blind Detection System Configuration\n")
                f.write("# This is an example configuration file with all available options\n")
                f.write("# \n")
                f.write("# Environment Variable Overrides:\n")
                f.write("# You can override any setting using environment variables with the prefix BLIND_DETECTION_\n")
                f.write("# Examples:\n")
                f.write("#   BLIND_DETECTION_MODEL=yolov8s.pt\n")
                f.write("#   BLIND_DETECTION_CONFIDENCE=0.7\n")
                f.write("#   BLIND_DETECTION_DEVICE=cuda\n")
                f.write("#   BLIND_DETECTION_TARGET_CLASSES=person,chair,table\n")
                f.write("# \n")
                f.write("# Configuration File Formats:\n")
                f.write("# Supported formats: .yaml, .yml, .json\n")
                f.write("# YAML format is recommended for human readability\n")
                f.write("# JSON format is supported for programmatic generation\n")
                f.write("\n")
                
                f.write("# Model settings\n")
                f.write("# model_name: YOLOv8 model file\n")
                f.write("#   Options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (most accurate)\n")
                f.write("#   Custom models: You can also use custom trained .pt files\n")
                f.write("#   Environment: BLIND_DETECTION_MODEL\n")
                f.write(f"model_name: '{self.config.model_name}'\n")
                f.write("\n")
                f.write("# confidence_threshold: Minimum confidence for detections (0.0-1.0)\n")
                f.write("#   Higher values = fewer but more confident detections\n")
                f.write("#   Lower values = more detections but potentially more false positives\n")
                f.write("#   Recommended range: 0.3-0.7 for most use cases\n")
                f.write("#   Environment: BLIND_DETECTION_CONFIDENCE\n")
                f.write(f"confidence_threshold: {self.config.confidence_threshold}\n")
                f.write("\n")
                f.write("# device: Processing device selection\n")
                f.write("#   Options: 'auto' (detect best), 'cpu' (force CPU), 'cuda' (force GPU)\n")
                f.write("#   'auto' will use GPU if available, otherwise CPU\n")
                f.write("#   'cuda' requires NVIDIA GPU with CUDA support\n")
                f.write("#   Environment: BLIND_DETECTION_DEVICE\n")
                f.write(f"device: '{self.config.device}'\n\n")
                
                f.write("# Audio settings\n")
                f.write("# voice_rate: Speech rate in words per minute (50-400)\n")
                f.write("#   Slower rates (100-150) for better comprehension\n")
                f.write("#   Faster rates (250-350) for experienced users\n")
                f.write("#   Environment: BLIND_DETECTION_VOICE_RATE\n")
                f.write(f"voice_rate: {self.config.voice_rate}\n")
                f.write("\n")
                f.write("# voice_volume: Speech volume level (0.0-1.0)\n")
                f.write("#   1.0 = maximum volume, 0.0 = muted\n")
                f.write("#   Recommended: 0.8-1.0 for clear audibility\n")
                f.write("#   Environment: BLIND_DETECTION_VOICE_VOLUME\n")
                f.write(f"voice_volume: {self.config.voice_volume}\n")
                f.write("\n")
                f.write("# announcement_cooldown: Seconds between repeated announcements for same object\n")
                f.write("#   Prevents audio spam for persistent objects\n")
                f.write("#   Lower values = more frequent updates\n")
                f.write("#   Higher values = less repetitive but potentially missed changes\n")
                f.write("#   Environment: BLIND_DETECTION_COOLDOWN\n")
                f.write(f"announcement_cooldown: {self.config.announcement_cooldown}\n\n")
                
                f.write("# Detection settings - Objects to detect and announce\n")
                f.write("# target_classes: List of object classes to detect\n")
                f.write("#   Environment: BLIND_DETECTION_TARGET_CLASSES (comma-separated)\n")
                f.write("#   Common navigation obstacles:\n")
                f.write("#     person, chair, table, sofa, bed, toilet, tv, laptop, mouse, keyboard,\n")
                f.write("#     cell phone, book, clock, scissors, teddy bear, hair drier, toothbrush\n")
                f.write("#   Additional household items:\n")
                f.write("#     bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich,\n")
                f.write("#     orange, broccoli, carrot, hot dog, pizza, donut, cake, potted plant, vase,\n")
                f.write("#     backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,\n")
                f.write("#     sports ball, kite, baseball bat, baseball glove, skateboard, surfboard\n")
                f.write("#   Vehicle related:\n")
                f.write("#     bicycle, car, motorcycle, airplane, bus, train, truck, boat\n")
                f.write("#   Animals:\n")
                f.write("#     bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe\n")
                f.write("target_classes:\n")
                for cls in self.config.target_classes:
                    f.write(f"  - '{cls}'\n")
                f.write("\n")
                
                f.write("# Camera settings\n")
                f.write("# camera_index: Camera device index (0 for default camera, 1+ for additional cameras)\n")
                f.write("#   Use 0 for built-in webcam, 1+ for external USB cameras\n")
                f.write("#   Environment: BLIND_DETECTION_CAMERA_INDEX\n")
                f.write(f"camera_index: {self.config.camera_index}\n")
                f.write("\n")
                f.write("# frame_width: Frame width in pixels\n")
                f.write("#   Lower resolution = faster processing, higher resolution = better accuracy\n")
                f.write("#   Recommended values: 320 (fast), 640 (balanced), 1280 (accurate), 1920 (high quality)\n")
                f.write("#   Environment: BLIND_DETECTION_FRAME_WIDTH\n")
                f.write(f"frame_width: {self.config.frame_width}\n")
                f.write("\n")
                f.write("# frame_height: Frame height in pixels\n")
                f.write("#   Should maintain aspect ratio with frame_width\n")
                f.write("#   Common ratios: 4:3 (640x480), 16:9 (1280x720, 1920x1080)\n")
                f.write("#   Environment: BLIND_DETECTION_FRAME_HEIGHT\n")
                f.write(f"frame_height: {self.config.frame_height}\n\n")
                
                f.write("# Spatial analysis settings\n")
                f.write("# distance_thresholds: Define close/medium/far based on object bounding box size\n")
                f.write("#   Values represent the fraction of frame area occupied by the object\n")
                f.write("#   Larger objects appear closer, smaller objects appear farther\n")
                f.write("#   Adjust these values based on your camera setup and room size\n")
                f.write("distance_thresholds:\n")
                f.write(f"  close: {self.config.distance_thresholds['close']}    # bbox area > 30% of frame (immediate attention required)\n")
                f.write(f"  medium: {self.config.distance_thresholds['medium']}   # bbox area 10-30% of frame (moderate attention)\n")
                f.write(f"  far: {self.config.distance_thresholds['far']}      # bbox area < 10% of frame (background awareness)\n")
                f.write("\n")
                f.write("# Advanced Configuration Notes:\n")
                f.write("# \n")
                f.write("# Configuration validation:\n")
                f.write("# The system automatically validates all configuration values on startup.\n")
                f.write("# Invalid values will cause the system to fall back to defaults with warnings.\n")
                f.write("# \n")
                f.write("# Hot reloading:\n")
                f.write("# The system can detect changes to this configuration file and reload automatically\n")
                f.write("# when config watching is enabled in the application.\n")
                f.write("# \n")
                f.write("# Performance tuning:\n")
                f.write("# - Use yolov8n.pt for fastest processing on slower hardware\n")
                f.write("# - Use yolov8s.pt or yolov8m.pt for balanced performance\n")
                f.write("# - Use yolov8l.pt or yolov8x.pt for highest accuracy on powerful hardware\n")
                f.write("# - Lower frame resolution for faster processing\n")
                f.write("# - Higher confidence threshold to reduce false positives\n")
                f.write("# \n")
                f.write("# Accessibility considerations:\n")
                f.write("# - Adjust voice_rate based on user preference and comprehension\n")
                f.write("# - Set appropriate announcement_cooldown to avoid audio overload\n")
                f.write("# - Customize target_classes based on user's environment and needs\n")
                f.write("# - Fine-tune distance_thresholds based on room size and camera position\n")
            
            print(f"Enhanced example configuration created at {output_path}")
            self.logger.info(f"Enhanced example configuration created at {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating example config: {e}")
            print(f"Error creating example config: {e}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = Config()
        self.logger.info("Configuration reset to defaults")
    
    def update_config(self, **kwargs) -> bool:
        """Update configuration with new values and validate."""
        try:
            # Store original config in case validation fails
            original_config = Config(
                model_name=self.config.model_name,
                confidence_threshold=self.config.confidence_threshold,
                device=self.config.device,
                voice_rate=self.config.voice_rate,
                voice_volume=self.config.voice_volume,
                announcement_cooldown=self.config.announcement_cooldown,
                target_classes=self.config.target_classes.copy(),
                camera_index=self.config.camera_index,
                frame_width=self.config.frame_width,
                frame_height=self.config.frame_height,
                distance_thresholds=self.config.distance_thresholds.copy()
            )
            
            # Update with new values
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    raise ValueError(f"Unknown configuration parameter: {key}")
            
            # Validate updated configuration
            if self.validate_config():
                self.logger.info(f"Configuration updated: {kwargs}")
                return True
            else:
                # Restore original configuration if validation fails
                self.config = original_config
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            print(f"Error updating configuration: {e}")
            return False
    
    def get_config_summary(self) -> str:
        """Get a human-readable summary of current configuration."""
        summary = []
        summary.append("=== Current Configuration ===")
        summary.append(f"Model: {self.config.model_name} (confidence: {self.config.confidence_threshold})")
        summary.append(f"Device: {self.config.device}")
        summary.append(f"Audio: Rate={self.config.voice_rate}wpm, Volume={self.config.voice_volume}")
        summary.append(f"Camera: Index={self.config.camera_index}, Resolution={self.config.frame_width}x{self.config.frame_height}")
        summary.append(f"Target Classes: {len(self.config.target_classes)} objects")
        summary.append(f"Distance Thresholds: Close>{self.config.distance_thresholds['close']}, Medium>{self.config.distance_thresholds['medium']}")
        return "\n".join(summary)
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'BLIND_DETECTION_MODEL': 'model_name',
            'BLIND_DETECTION_CONFIDENCE': 'confidence_threshold',
            'BLIND_DETECTION_DEVICE': 'device',
            'BLIND_DETECTION_VOICE_RATE': 'voice_rate',
            'BLIND_DETECTION_VOICE_VOLUME': 'voice_volume',
            'BLIND_DETECTION_COOLDOWN': 'announcement_cooldown',
            'BLIND_DETECTION_CAMERA_INDEX': 'camera_index',
            'BLIND_DETECTION_FRAME_WIDTH': 'frame_width',
            'BLIND_DETECTION_FRAME_HEIGHT': 'frame_height'
        }
        
        for env_var, config_attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Type conversion based on attribute type
                    current_value = getattr(self.config, config_attr)
                    if isinstance(current_value, bool):
                        converted_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        converted_value = int(env_value)
                    elif isinstance(current_value, float):
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                    
                    setattr(self.config, config_attr, converted_value)
                    self.logger.info(f"Applied environment override: {env_var}={env_value}")
                    
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")
        
        # Handle target classes from environment (comma-separated)
        target_classes_env = os.getenv('BLIND_DETECTION_TARGET_CLASSES')
        if target_classes_env:
            try:
                self.config.target_classes = [cls.strip() for cls in target_classes_env.split(',')]
                self.logger.info(f"Applied environment override for target_classes")
            except Exception as e:
                self.logger.warning(f"Invalid target classes environment variable: {e}")
    
    def check_config_changes(self) -> bool:
        """Check if configuration file has been modified since last load."""
        if not os.path.exists(self.config_path):
            return False
        
        try:
            current_modified = os.path.getmtime(self.config_path)
            return current_modified > self._last_modified
        except OSError:
            return False
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if file has changed. Returns True if reloaded."""
        if self.check_config_changes():
            try:
                old_config = self.get_config_summary()
                self.load_config()
                new_config = self.get_config_summary()
                
                if old_config != new_config:
                    self.logger.info("Configuration reloaded due to file changes")
                    print("Configuration reloaded due to file changes")
                    return True
            except Exception as e:
                self.logger.error(f"Error reloading configuration: {e}")
        
        return False
    
    def enable_config_watching(self) -> None:
        """Enable automatic configuration file watching."""
        self._watch_config = True
        self.logger.info("Configuration file watching enabled")
    
    def disable_config_watching(self) -> None:
        """Disable automatic configuration file watching."""
        self._watch_config = False
        self.logger.info("Configuration file watching disabled")
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for validation and documentation."""
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "YOLOv8 model file name",
                    "examples": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                    "default": "yolov8n.pt"
                },
                "confidence_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence for object detections",
                    "default": 0.5
                },
                "device": {
                    "type": "string",
                    "enum": ["auto", "cpu", "cuda"],
                    "description": "Processing device selection",
                    "default": "auto"
                },
                "voice_rate": {
                    "type": "integer",
                    "minimum": 50,
                    "maximum": 400,
                    "description": "Speech rate in words per minute",
                    "default": 200
                },
                "voice_volume": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Speech volume level",
                    "default": 0.9
                },
                "announcement_cooldown": {
                    "type": "number",
                    "minimum": 0.0,
                    "description": "Seconds between repeated announcements",
                    "default": 2.0
                },
                "target_classes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "List of object classes to detect and announce",
                    "default": ["person", "chair", "table", "sofa", "bed"]
                },
                "camera_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Camera device index",
                    "default": 0
                },
                "frame_width": {
                    "type": "integer",
                    "minimum": 320,
                    "description": "Frame width in pixels",
                    "default": 640
                },
                "frame_height": {
                    "type": "integer",
                    "minimum": 240,
                    "description": "Frame height in pixels",
                    "default": 480
                },
                "distance_thresholds": {
                    "type": "object",
                    "properties": {
                        "close": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "medium": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "far": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    },
                    "required": ["close", "medium", "far"],
                    "description": "Distance thresholds based on object size",
                    "default": {"close": 0.3, "medium": 0.1, "far": 0.0}
                }
            },
            "required": ["model_name", "confidence_threshold", "device", "target_classes"]
        }
    
    def validate_against_schema(self) -> List[str]:
        """Validate configuration against schema and return list of errors."""
        errors = []
        schema = self.get_config_schema()
        
        # Basic validation against schema
        config_dict = asdict(self.config)
        
        for prop, prop_schema in schema["properties"].items():
            if prop not in config_dict:
                if prop in schema.get("required", []):
                    errors.append(f"Required property '{prop}' is missing")
                continue
            
            value = config_dict[prop]
            prop_type = prop_schema.get("type")
            
            # Type validation
            if prop_type == "string" and not isinstance(value, str):
                errors.append(f"Property '{prop}' must be a string, got {type(value).__name__}")
            elif prop_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Property '{prop}' must be a number, got {type(value).__name__}")
            elif prop_type == "integer" and not isinstance(value, int):
                errors.append(f"Property '{prop}' must be an integer, got {type(value).__name__}")
            elif prop_type == "array" and not isinstance(value, list):
                errors.append(f"Property '{prop}' must be an array, got {type(value).__name__}")
            elif prop_type == "object" and not isinstance(value, dict):
                errors.append(f"Property '{prop}' must be an object, got {type(value).__name__}")
            
            # Range validation
            if isinstance(value, (int, float)):
                if "minimum" in prop_schema and value < prop_schema["minimum"]:
                    errors.append(f"Property '{prop}' must be >= {prop_schema['minimum']}, got {value}")
                if "maximum" in prop_schema and value > prop_schema["maximum"]:
                    errors.append(f"Property '{prop}' must be <= {prop_schema['maximum']}, got {value}")
            
            # Enum validation
            if "enum" in prop_schema and value not in prop_schema["enum"]:
                errors.append(f"Property '{prop}' must be one of {prop_schema['enum']}, got '{value}'")
            
            # Array validation
            if prop_type == "array" and isinstance(value, list):
                if "minItems" in prop_schema and len(value) < prop_schema["minItems"]:
                    errors.append(f"Property '{prop}' must have at least {prop_schema['minItems']} items")
        
        return errors