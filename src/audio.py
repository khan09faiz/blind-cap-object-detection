"""
Audio management system with intelligent speech synthesis and announcement logic.
"""

import pyttsx3
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from .detector import Detection
from .navigation import NavigationContext, WarningLevel

# Set up logging
logger = logging.getLogger(__name__)


class Priority(Enum):
    """Speech priority levels for announcement management."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class AnnouncementRecord:
    """Record of previous announcements for repetition prevention."""
    message: str
    timestamp: float
    detection_key: str  # Unique key for the detection (class_name + position)


class AudioManager:
    """Intelligent audio management system with pyttsx3 integration."""
    
    # Make Priority enum accessible as class attribute
    Priority = Priority
    
    def __init__(self, voice_rate: int = 200, voice_volume: float = 0.9, 
                 announcement_cooldown: float = 2.0):
        """
        Initialize the AudioManager.
        
        Args:
            voice_rate: Speech rate (words per minute)
            voice_volume: Voice volume (0.0 to 1.0)
            announcement_cooldown: Minimum time between similar announcements
        """
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume
        self.announcement_cooldown = announcement_cooldown
        
        # TTS engine
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        
        # Announcement tracking
        self.last_announcements: Dict[str, AnnouncementRecord] = {}
        self.current_speech_priority = Priority.LOW
        self.is_speaking = False
        
        # Initialize TTS engine
        self.initialize_tts()
    
    def initialize_tts(self) -> bool:
        """
        Initialize the text-to-speech engine with configuration.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice properties
            self.tts_engine.setProperty('rate', self.voice_rate)
            self.tts_engine.setProperty('volume', self.voice_volume)
            
            # Try to set a clear voice (prefer female voices for better clarity)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Look for a female voice first, fallback to first available
                female_voice = None
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        female_voice = voice
                        break
                
                if female_voice:
                    self.tts_engine.setProperty('voice', female_voice.id)
                    logger.info(f"Using voice: {female_voice.name}")
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
                    logger.info(f"Using default voice: {voices[0].name}")
            
            logger.info("TTS engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            return False
    
    def announce(self, message: str, priority: Priority = Priority.NORMAL) -> None:
        """
        Announce a message with priority-based speech interruption.
        
        Args:
            message: Text message to announce
            priority: Priority level for the announcement
        """
        if not self.tts_engine:
            logger.warning("TTS engine not available, printing to console")
            print(f"[AUDIO] {message}")
            return
        
        with self.tts_lock:
            # Handle priority-based interruption
            if priority.value > self.current_speech_priority.value and self.is_speaking:
                try:
                    self.tts_engine.stop()
                    logger.info(f"Interrupted speech for {priority.name} priority message")
                except Exception as e:
                    logger.warning(f"Failed to interrupt speech: {e}")
            
            # Update current priority and speaking status
            self.current_speech_priority = priority
            self.is_speaking = True
            
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
                logger.debug(f"Announced: {message}")
            except Exception as e:
                logger.error(f"Failed to announce message: {e}")
                print(f"[AUDIO] {message}")  # Fallback to console
            finally:
                self.is_speaking = False
                self.current_speech_priority = Priority.LOW
    
    def should_announce(self, detection: Detection, position: str, distance: str) -> bool:
        """
        Determine if a detection should be announced based on cooldown logic.
        
        Args:
            detection: Detection object
            position: Spatial position (left/center/right)
            distance: Distance category (close/medium/far)
            
        Returns:
            bool: True if announcement should be made
        """
        current_time = time.time()
        detection_key = f"{detection.class_name}_{position}_{distance}"
        
        # Check if we have a recent announcement for this detection
        if detection_key in self.last_announcements:
            last_record = self.last_announcements[detection_key]
            time_since_last = current_time - last_record.timestamp
            
            # Apply cooldown logic
            if time_since_last < self.announcement_cooldown:
                return False
        
        # Update the announcement record
        self.last_announcements[detection_key] = AnnouncementRecord(
            message="",  # Will be filled when actually announcing
            timestamp=current_time,
            detection_key=detection_key
        )
        
        return True
    
    def generate_spatial_description(self, detection: Detection, position: str, distance: str) -> str:
        """
        Generate natural language spatial description for a detection.
        
        Args:
            detection: Detection object
            position: Spatial position (left/center/right)
            distance: Distance category (close/medium/far)
            
        Returns:
            str: Natural language description
        """
        object_name = detection.class_name.replace('_', ' ')
        
        # Generate position description
        if position == "center":
            position_desc = "ahead of you"
        else:
            position_desc = f"to your {position}"
        
        # Generate distance-based action guidance
        action_guidance = ""
        if distance == "close":
            if position == "center":
                action_guidance = ", move cautiously"
            else:
                action_guidance = ", be aware"
        elif distance == "medium":
            if position == "center":
                action_guidance = ", path partially blocked"
        
        # Construct the message
        message = f"{object_name} {position_desc}"
        if distance == "close":
            message = f"{object_name} close {position_desc}{action_guidance}"
        elif distance == "medium":
            message = f"{object_name} {position_desc}{action_guidance}"
        
        return message
    
    def generate_navigation_guidance(self, detections: List[Tuple[Detection, str, str]]) -> Optional[str]:
        """
        Generate contextual navigation guidance based on multiple detections.
        
        Args:
            detections: List of (Detection, position, distance) tuples
            
        Returns:
            Optional[str]: Navigation guidance message or None
        """
        if not detections:
            return None
        
        # Categorize detections by position and distance
        center_close = []
        center_other = []
        side_close = []
        
        for detection, position, distance in detections:
            if position == "center" and distance == "close":
                center_close.append(detection)
            elif position == "center":
                center_other.append(detection)
            elif distance == "close":
                side_close.append(detection)
        
        # Generate guidance based on situation
        if center_close:
            if len(center_close) == 1:
                obj_name = center_close[0].class_name.replace('_', ' ')
                return f"Stop, {obj_name} directly ahead"
            else:
                return "Stop, multiple objects blocking path"
        
        if center_other:
            if len(center_other) == 1:
                obj_name = center_other[0].class_name.replace('_', ' ')
                return f"Caution, {obj_name} ahead"
            else:
                return "Caution, objects ahead"
        
        if side_close:
            if len(side_close) == 1:
                obj_name = side_close[0].class_name.replace('_', ' ')
                return f"Be aware, {obj_name} nearby"
        
        return None
    
    def announce_detection(self, detection: Detection, position: str, distance: str) -> None:
        """
        Announce a single detection with appropriate priority and messaging.
        
        Args:
            detection: Detection object
            position: Spatial position
            distance: Distance category
        """
        if not self.should_announce(detection, position, distance):
            return
        
        # Determine priority based on position and distance
        priority = Priority.NORMAL
        if position == "center" and distance == "close":
            priority = Priority.URGENT
        elif position == "center" and distance == "medium":
            priority = Priority.HIGH
        elif distance == "close":
            priority = Priority.HIGH
        
        # Generate and announce the message
        message = self.generate_spatial_description(detection, position, distance)
        self.announce(message, priority)
        
        # Update the announcement record with the actual message
        detection_key = f"{detection.class_name}_{position}_{distance}"
        if detection_key in self.last_announcements:
            self.last_announcements[detection_key].message = message
    
    def announce_multiple_detections(self, detections: List[Tuple[Detection, str, str]]) -> None:
        """
        Announce multiple detections with intelligent prioritization.
        
        Args:
            detections: List of (Detection, position, distance) tuples
        """
        if not detections:
            return
        
        # Sort by priority (center + close first)
        def detection_priority(item):
            detection, position, distance = item
            score = 0
            if position == "center":
                score += 10
            if distance == "close":
                score += 5
            elif distance == "medium":
                score += 2
            return score
        
        sorted_detections = sorted(detections, key=detection_priority, reverse=True)
        
        # Announce the highest priority detection
        if sorted_detections:
            detection, position, distance = sorted_detections[0]
            self.announce_detection(detection, position, distance)
        
        # If there are multiple high-priority detections, provide summary guidance
        if len(sorted_detections) > 1:
            guidance = self.generate_navigation_guidance(sorted_detections)
            if guidance:
                # Small delay to avoid overlapping speech
                threading.Timer(1.0, lambda: self.announce(guidance, Priority.HIGH)).start()
    
    def announce_navigation_guidance(self, guidance_messages: List[str]) -> None:
        """
        Announce navigation guidance messages with appropriate priorities.
        
        Args:
            guidance_messages: List of guidance messages in priority order
        """
        if not guidance_messages:
            return
        
        # Announce the highest priority message first
        primary_message = guidance_messages[0]
        
        # Determine priority based on message content
        priority = Priority.NORMAL
        if any(keyword in primary_message.lower() for keyword in ['stop', 'urgent', 'hazard', 'danger']):
            priority = Priority.URGENT
        elif any(keyword in primary_message.lower() for keyword in ['warning', 'caution', 'close']):
            priority = Priority.HIGH
        elif 'path clear' in primary_message.lower():
            priority = Priority.NORMAL
        
        self.announce(primary_message, priority)
        
        # If there are additional messages, announce them with a delay
        if len(guidance_messages) > 1:
            for i, message in enumerate(guidance_messages[1:], 1):
                delay = i * 1.5  # Stagger additional messages
                threading.Timer(delay, lambda msg=message: self.announce(msg, Priority.NORMAL)).start()
    
    def announce_context_based_guidance(self, contexts: List[NavigationContext]) -> None:
        """
        Announce guidance based on navigation contexts.
        
        Args:
            contexts: List of navigation contexts
        """
        if not contexts:
            return
        
        # Sort contexts by warning level
        sorted_contexts = sorted(contexts, key=lambda ctx: ctx.warning_level.value, reverse=True)
        
        # Announce the highest priority context
        primary_context = sorted_contexts[0]
        
        # Map warning levels to audio priorities
        priority_mapping = {
            WarningLevel.URGENT: Priority.URGENT,
            WarningLevel.WARNING: Priority.HIGH,
            WarningLevel.CAUTION: Priority.NORMAL,
            WarningLevel.INFO: Priority.LOW
        }
        
        audio_priority = priority_mapping.get(primary_context.warning_level, Priority.NORMAL)
        
        # Create comprehensive message
        if primary_context.warning_level in [WarningLevel.URGENT, WarningLevel.WARNING]:
            message = f"{primary_context.guidance_message}. {primary_context.action_suggestion}"
        else:
            message = primary_context.guidance_message
        
        self.announce(message, audio_priority)
    
    def announce_escalating_warning(self, warning_message: str) -> None:
        """
        Announce escalating warning with high priority.
        
        Args:
            warning_message: Escalating warning message
        """
        self.announce(warning_message, Priority.HIGH)
    
    def announce_path_status(self, is_clear: bool, previous_state: bool = None) -> None:
        """
        Announce path status changes.
        
        Args:
            is_clear: Current path clear status
            previous_state: Previous path clear status
        """
        # Only announce if state changed from blocked to clear
        if is_clear and previous_state is False:
            self.announce("Path clear", Priority.NORMAL)
        elif not is_clear and previous_state is True:
            self.announce("Path blocked", Priority.HIGH)
    
    def should_announce_object_category(self, detection: Detection, category_name: str, 
                                      position: str, distance: str) -> bool:
        """
        Determine if object should be announced based on category-specific rules.
        
        Args:
            detection: Detection object
            category_name: Object category name
            position: Spatial position
            distance: Distance category
            
        Returns:
            bool: True if should announce
        """
        # Create category-specific detection key
        detection_key = f"{category_name}_{detection.class_name}_{position}_{distance}"
        
        # Use existing cooldown logic but with category-aware keys
        current_time = time.time()
        
        if detection_key in self.last_announcements:
            last_record = self.last_announcements[detection_key]
            time_since_last = current_time - last_record.timestamp
            
            # Category-specific cooldown periods
            category_cooldowns = {
                'person': 3.0,      # Longer cooldown for people
                'furniture': 4.0,   # Longer cooldown for furniture
                'hazard': 1.0,      # Shorter cooldown for hazards
                'vehicle': 2.0,     # Medium cooldown for vehicles
            }
            
            cooldown = category_cooldowns.get(category_name, self.announcement_cooldown)
            
            if time_since_last < cooldown:
                return False
        
        # Update announcement record
        self.last_announcements[detection_key] = AnnouncementRecord(
            message="",
            timestamp=current_time,
            detection_key=detection_key
        )
        
        return True
    
    def generate_safety_warning_message(self, detection: Detection, position: str, distance: str) -> str:
        """
        Generate safety-focused warning message for dangerous objects.
        
        Args:
            detection: Detection object
            position: Spatial position
            distance: Distance category
            
        Returns:
            str: Safety warning message
        """
        object_name = detection.class_name.replace('_', ' ')
        
        # Immediate danger objects
        dangerous_objects = ['scissors', 'knife', 'car', 'truck', 'bus', 'motorcycle']
        
        if detection.class_name in dangerous_objects:
            if distance == 'close':
                if position == 'center':
                    return f"Danger! {object_name} directly ahead, stop immediately"
                else:
                    return f"Warning! {object_name} {position}, very close"
            else:
                return f"Caution: {object_name} detected {position}"
        
        # Person-specific messaging
        elif detection.class_name == 'person':
            if distance == 'close' and position == 'center':
                return "Person ahead, stop and wait"
            elif distance == 'close':
                return f"Person {position}, nearby"
            else:
                return f"Person {position}"
        
        # Default safety message
        else:
            if distance == 'close' and position == 'center':
                return f"{object_name} blocking path, navigate around"
            else:
                return f"{object_name} {position}"
    
    def cleanup(self) -> None:
        """Clean up TTS resources."""
        with self.tts_lock:
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except Exception as e:
                    logger.warning(f"Error stopping TTS engine: {e}")
                finally:
                    self.tts_engine = None
        
        logger.info("AudioManager cleanup completed")