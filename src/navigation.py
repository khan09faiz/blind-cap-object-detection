"""
Navigation guidance system for contextual safety guidance and object-specific messaging.
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import time
import logging
from .detector import Detection

logger = logging.getLogger(__name__)


class ObjectCategory(Enum):
    """Categories of objects for different guidance strategies."""
    PERSON = "person"
    FURNITURE = "furniture"
    HAZARD = "hazard"
    VEHICLE = "vehicle"
    ELECTRONIC = "electronic"
    FOOD = "food"
    UNKNOWN = "unknown"


class WarningLevel(Enum):
    """Warning levels for escalating guidance."""
    INFO = 1
    CAUTION = 2
    WARNING = 3
    URGENT = 4


@dataclass
class NavigationContext:
    """Context information for navigation guidance."""
    detection: Detection
    position: str  # left/center/right
    distance: str  # close/medium/far
    category: ObjectCategory
    warning_level: WarningLevel
    guidance_message: str
    action_suggestion: str


class NavigationGuidanceSystem:
    """Contextual safety guidance system with object-specific messaging."""
    
    def __init__(self):
        """Initialize the navigation guidance system."""
        # Object categorization mapping
        self.object_categories = {
            # People
            'person': ObjectCategory.PERSON,
            
            # Furniture and obstacles
            'chair': ObjectCategory.FURNITURE,
            'table': ObjectCategory.FURNITURE,
            'sofa': ObjectCategory.FURNITURE,
            'bed': ObjectCategory.FURNITURE,
            'toilet': ObjectCategory.FURNITURE,
            'bench': ObjectCategory.FURNITURE,
            'couch': ObjectCategory.FURNITURE,
            'desk': ObjectCategory.FURNITURE,
            'cabinet': ObjectCategory.FURNITURE,
            'shelf': ObjectCategory.FURNITURE,
            
            # Hazardous objects
            'scissors': ObjectCategory.HAZARD,
            'knife': ObjectCategory.HAZARD,
            'fire hydrant': ObjectCategory.HAZARD,
            'stop sign': ObjectCategory.HAZARD,
            
            # Vehicles
            'car': ObjectCategory.VEHICLE,
            'truck': ObjectCategory.VEHICLE,
            'bus': ObjectCategory.VEHICLE,
            'motorcycle': ObjectCategory.VEHICLE,
            'bicycle': ObjectCategory.VEHICLE,
            
            # Electronics
            'tv': ObjectCategory.ELECTRONIC,
            'laptop': ObjectCategory.ELECTRONIC,
            'cell phone': ObjectCategory.ELECTRONIC,
            'keyboard': ObjectCategory.ELECTRONIC,
            'mouse': ObjectCategory.ELECTRONIC,
            'remote': ObjectCategory.ELECTRONIC,
            
            # Food items
            'banana': ObjectCategory.FOOD,
            'apple': ObjectCategory.FOOD,
            'orange': ObjectCategory.FOOD,
            'bottle': ObjectCategory.FOOD,
            'cup': ObjectCategory.FOOD,
            'bowl': ObjectCategory.FOOD,
        }
        
        # Warning level thresholds based on category, position, and distance
        self.warning_thresholds = {
            ObjectCategory.PERSON: {
                ('center', 'close'): WarningLevel.WARNING,
                ('center', 'medium'): WarningLevel.CAUTION,
                ('left', 'close'): WarningLevel.CAUTION,
                ('right', 'close'): WarningLevel.CAUTION,
            },
            ObjectCategory.FURNITURE: {
                ('center', 'close'): WarningLevel.WARNING,
                ('center', 'medium'): WarningLevel.CAUTION,
                ('left', 'close'): WarningLevel.CAUTION,
                ('right', 'close'): WarningLevel.INFO,
            },
            ObjectCategory.HAZARD: {
                ('center', 'close'): WarningLevel.URGENT,
                ('center', 'medium'): WarningLevel.WARNING,
                ('left', 'close'): WarningLevel.WARNING,
                ('right', 'close'): WarningLevel.WARNING,
                ('left', 'medium'): WarningLevel.WARNING,
                ('right', 'medium'): WarningLevel.WARNING,
            },
            ObjectCategory.VEHICLE: {
                ('center', 'close'): WarningLevel.URGENT,
                ('center', 'medium'): WarningLevel.WARNING,
                ('left', 'close'): WarningLevel.WARNING,
                ('right', 'close'): WarningLevel.WARNING,
            },
        }
        
        # Track path state for "path clear" announcements
        self.path_was_blocked = False
        self.last_path_clear_time = 0
        self.path_clear_cooldown = 5.0  # seconds
    
    def categorize_object(self, class_name: str) -> ObjectCategory:
        """
        Categorize an object based on its class name.
        
        Args:
            class_name: Object class name
            
        Returns:
            ObjectCategory: Category of the object
        """
        return self.object_categories.get(class_name.lower(), ObjectCategory.UNKNOWN)
    
    def determine_warning_level(self, detection: Detection, position: str, distance: str) -> WarningLevel:
        """
        Determine warning level based on object category, position, and distance.
        
        Args:
            detection: Detection object
            position: Spatial position (left/center/right)
            distance: Distance category (close/medium/far)
            
        Returns:
            WarningLevel: Appropriate warning level
        """
        category = self.categorize_object(detection.class_name)
        
        # Get warning level from thresholds
        if category in self.warning_thresholds:
            threshold_key = (position, distance)
            warning_level = self.warning_thresholds[category].get(threshold_key, WarningLevel.INFO)
        else:
            # Default warning levels for unknown categories
            if position == 'center' and distance == 'close':
                warning_level = WarningLevel.WARNING
            elif distance == 'close':
                warning_level = WarningLevel.CAUTION
            else:
                warning_level = WarningLevel.INFO
        
        return warning_level
    
    def generate_object_specific_message(self, detection: Detection, position: str, distance: str, 
                                       category: ObjectCategory) -> str:
        """
        Generate object-specific guidance message.
        
        Args:
            detection: Detection object
            position: Spatial position
            distance: Distance category
            category: Object category
            
        Returns:
            str: Object-specific guidance message
        """
        object_name = detection.class_name.replace('_', ' ')
        
        # Position descriptions
        position_phrases = {
            'left': 'to your left',
            'center': 'ahead of you',
            'right': 'to your right'
        }
        position_text = position_phrases.get(position, position)
        
        # Category-specific messaging
        if category == ObjectCategory.PERSON:
            if position == 'center' and distance == 'close':
                return f"Person directly ahead, very close"
            elif distance == 'close':
                return f"Person {position_text}, nearby"
            else:
                return f"Person {position_text}"
        
        elif category == ObjectCategory.FURNITURE:
            if position == 'center' and distance == 'close':
                return f"{object_name} blocking path ahead"
            elif position == 'center':
                return f"{object_name} directly ahead"
            else:
                return f"{object_name} {position_text}"
        
        elif category == ObjectCategory.HAZARD:
            if distance == 'close':
                if position == 'center':
                    return f"Hazard: {object_name} directly ahead, very close"
                else:
                    return f"Hazard: {object_name} {position_text}, very close"
            else:
                if position == 'center':
                    return f"Hazard: {object_name} directly ahead"
                else:
                    return f"Hazard: {object_name} {position_text}"
        
        elif category == ObjectCategory.VEHICLE:
            if distance == 'close':
                return f"Vehicle: {object_name} {position_text}, close"
            else:
                return f"Vehicle: {object_name} {position_text}"
        
        else:
            # Generic message for unknown categories
            if distance == 'close':
                return f"{object_name} {position_text}, close"
            else:
                return f"{object_name} {position_text}"
    
    def generate_action_suggestion(self, detection: Detection, position: str, distance: str, 
                                 category: ObjectCategory, warning_level: WarningLevel) -> str:
        """
        Generate action suggestion based on context.
        
        Args:
            detection: Detection object
            position: Spatial position
            distance: Distance category
            category: Object category
            warning_level: Warning level
            
        Returns:
            str: Action suggestion
        """
        if warning_level == WarningLevel.URGENT:
            return "Stop immediately"
        
        elif warning_level == WarningLevel.WARNING:
            if position == 'center':
                if category == ObjectCategory.PERSON:
                    return "Stop and wait"
                elif category == ObjectCategory.FURNITURE:
                    return "Navigate around obstacle"
                else:
                    return "Stop and assess"
            else:
                return "Move cautiously"
        
        elif warning_level == WarningLevel.CAUTION:
            if position == 'center':
                if category == ObjectCategory.PERSON:
                    return "Approach slowly"
                else:
                    return "Prepare to navigate around"
            else:
                if category == ObjectCategory.PERSON and distance == 'close':
                    return "Move cautiously"
                else:
                    return "Be aware"
        
        else:  # INFO level
            return "Continue with awareness"
    
    def create_navigation_context(self, detection: Detection, position: str, distance: str) -> NavigationContext:
        """
        Create comprehensive navigation context for a detection.
        
        Args:
            detection: Detection object
            position: Spatial position
            distance: Distance category
            
        Returns:
            NavigationContext: Complete navigation context
        """
        category = self.categorize_object(detection.class_name)
        warning_level = self.determine_warning_level(detection, position, distance)
        guidance_message = self.generate_object_specific_message(detection, position, distance, category)
        action_suggestion = self.generate_action_suggestion(detection, position, distance, category, warning_level)
        
        return NavigationContext(
            detection=detection,
            position=position,
            distance=distance,
            category=category,
            warning_level=warning_level,
            guidance_message=guidance_message,
            action_suggestion=action_suggestion
        )
    
    def generate_escalating_warning(self, contexts: List[NavigationContext], 
                                  previous_contexts: Optional[List[NavigationContext]] = None) -> Optional[str]:
        """
        Generate escalating warning messages for approaching objects.
        
        Args:
            contexts: Current navigation contexts
            previous_contexts: Previous frame contexts for comparison
            
        Returns:
            Optional[str]: Escalating warning message if needed
        """
        if not contexts or not previous_contexts:
            return None
        
        # Find objects that are getting closer
        for current_ctx in contexts:
            for prev_ctx in previous_contexts:
                # Match objects by class name and position
                if (current_ctx.detection.class_name == prev_ctx.detection.class_name and
                    current_ctx.position == prev_ctx.position):
                    
                    # Check if object moved from medium/far to close
                    if (prev_ctx.distance in ['medium', 'far'] and 
                        current_ctx.distance == 'close'):
                        
                        if current_ctx.category == ObjectCategory.PERSON:
                            return f"Person approaching {current_ctx.position}"
                        elif current_ctx.category == ObjectCategory.HAZARD:
                            return f"Warning: {current_ctx.detection.class_name} getting closer"
                        else:
                            return f"{current_ctx.detection.class_name} getting closer"
        
        return None
    
    def check_path_clear(self, contexts: List[NavigationContext]) -> Optional[str]:
        """
        Check if path is clear and generate appropriate message.
        
        Args:
            contexts: Current navigation contexts
            
        Returns:
            Optional[str]: Path clear message if appropriate
        """
        current_time = time.time()
        
        # Check if there are any center objects that are close or medium distance
        center_obstacles = [ctx for ctx in contexts 
                          if ctx.position == 'center' and ctx.distance in ['close', 'medium']]
        
        path_currently_blocked = len(center_obstacles) > 0
        
        # If path was blocked but now is clear, announce it
        if self.path_was_blocked and not path_currently_blocked:
            if current_time - self.last_path_clear_time > self.path_clear_cooldown:
                self.last_path_clear_time = current_time
                self.path_was_blocked = False
                return "Path clear"
        
        # Update path state
        self.path_was_blocked = path_currently_blocked
        
        return None
    
    def prioritize_contexts(self, contexts: List[NavigationContext]) -> List[NavigationContext]:
        """
        Prioritize navigation contexts by importance.
        
        Args:
            contexts: List of navigation contexts
            
        Returns:
            List[NavigationContext]: Sorted contexts by priority
        """
        def priority_score(ctx: NavigationContext) -> float:
            score = 0
            
            # Warning level priority
            score += ctx.warning_level.value * 10
            
            # Position priority (center is most important)
            if ctx.position == 'center':
                score += 5
            
            # Distance priority (closer is more important)
            distance_scores = {'close': 3, 'medium': 2, 'far': 1}
            score += distance_scores.get(ctx.distance, 1)
            
            # Category priority
            category_scores = {
                ObjectCategory.HAZARD: 4,
                ObjectCategory.VEHICLE: 3,
                ObjectCategory.PERSON: 2,
                ObjectCategory.FURNITURE: 1
            }
            score += category_scores.get(ctx.category, 0)
            
            # Confidence bonus
            score += ctx.detection.confidence * 0.5
            
            return score
        
        return sorted(contexts, key=priority_score, reverse=True)
    
    def generate_comprehensive_guidance(self, contexts: List[NavigationContext], 
                                      previous_contexts: Optional[List[NavigationContext]] = None) -> List[str]:
        """
        Generate comprehensive navigation guidance messages.
        
        Args:
            contexts: Current navigation contexts
            previous_contexts: Previous contexts for escalation detection
            
        Returns:
            List[str]: List of guidance messages in priority order
        """
        messages = []
        
        if not contexts:
            # Check for path clear message
            path_clear_msg = self.check_path_clear([])
            if path_clear_msg:
                messages.append(path_clear_msg)
            return messages
        
        # Prioritize contexts
        prioritized_contexts = self.prioritize_contexts(contexts)
        
        # Check for escalating warnings
        escalating_warning = self.generate_escalating_warning(contexts, previous_contexts)
        if escalating_warning:
            messages.append(escalating_warning)
        
        # Generate primary guidance message
        if prioritized_contexts:
            primary_ctx = prioritized_contexts[0]
            
            # Combine guidance message with action suggestion for urgent/warning cases
            if primary_ctx.warning_level in [WarningLevel.URGENT, WarningLevel.WARNING]:
                combined_message = f"{primary_ctx.guidance_message}. {primary_ctx.action_suggestion}"
                messages.append(combined_message)
            else:
                messages.append(primary_ctx.guidance_message)
        
        # Check for path clear message
        path_clear_msg = self.check_path_clear(contexts)
        if path_clear_msg:
            messages.append(path_clear_msg)
        
        return messages