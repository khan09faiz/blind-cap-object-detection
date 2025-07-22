"""
Spatial analysis module for calculating object positions and distances.
"""

from typing import List, Dict, Tuple, Optional
from .detector import Detection
from .navigation import NavigationGuidanceSystem, NavigationContext


class SpatialAnalyzer:
    """Analyze object positions and calculate spatial relationships."""
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """
        Initialize the spatial analyzer.
        
        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Initialize navigation guidance system
        self.navigation_system = NavigationGuidanceSystem()
        
        # Store previous contexts for escalation detection
        self.previous_contexts: Optional[List[NavigationContext]] = None
        
        # Define danger levels for different object classes
        self.danger_levels = {
            'person': 3,
            'car': 4,
            'truck': 4,
            'bus': 4,
            'motorcycle': 4,
            'bicycle': 3,
            'chair': 2,
            'table': 2,
            'sofa': 2,
            'bed': 1,
            'toilet': 1,
            'tv': 1,
            'laptop': 1,
            'book': 1,
            'scissors': 3,
            'knife': 4,
            'bottle': 2,
            'cup': 2,
            'bowl': 2,
            'banana': 1,
            'apple': 1,
            'orange': 1,
            'broccoli': 1,
            'carrot': 1,
            'hot dog': 1,
            'pizza': 1,
            'donut': 1,
            'cake': 1
        }
    
    def calculate_position(self, detection: Detection, frame_width: int = None) -> str:
        """
        Calculate object position (left/center/right).
        
        Args:
            detection: Detection object containing bounding box information
            frame_width: Width of the frame (optional, uses instance default if not provided)
            
        Returns:
            str: Position as 'left', 'center', or 'right'
        """
        if frame_width is None:
            frame_width = self.frame_width
            
        center_x = detection.center_point[0]
        
        # Define zones: left (0-33%), center (33-67%), right (67-100%)
        left_boundary = frame_width * 0.33
        right_boundary = frame_width * 0.67
        
        if center_x < left_boundary:
            return 'left'
        elif center_x > right_boundary:
            return 'right'
        else:
            return 'center'
    
    def estimate_distance(self, detection: Detection, frame_width: int = None, frame_height: int = None) -> str:
        """
        Estimate distance based on bounding box size (close/medium/far).
        
        Args:
            detection: Detection object containing bounding box information
            frame_width: Width of the frame (optional)
            frame_height: Height of the frame (optional)
            
        Returns:
            str: Distance category as 'close', 'medium', or 'far'
        """
        if frame_width is None:
            frame_width = self.frame_width
        if frame_height is None:
            frame_height = self.frame_height
            
        x1, y1, x2, y2 = detection.bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        
        frame_area = frame_width * frame_height
        area_ratio = bbox_area / frame_area
        
        # Distance estimation based on bounding box area relative to frame
        if area_ratio > 0.30:  # > 30% of frame
            return 'close'
        elif area_ratio > 0.10:  # 10-30% of frame
            return 'medium'
        else:  # < 10% of frame
            return 'far'
    
    def get_danger_level(self, class_name: str) -> int:
        """
        Get danger level for a given object class.
        
        Args:
            class_name: Name of the object class
            
        Returns:
            int: Danger level (1-4, where 4 is most dangerous)
        """
        return self.danger_levels.get(class_name.lower(), 2)  # Default to medium danger
    
    def prioritize_objects(self, detections: List[Detection]) -> List[Detection]:
        """
        Prioritize objects based on proximity and danger level.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List[Detection]: Sorted list of detections by priority (highest first)
        """
        def priority_score(detection: Detection) -> float:
            """Calculate priority score for a detection."""
            # Get distance category and convert to numeric score
            distance = self.estimate_distance(detection)
            distance_scores = {'close': 3.0, 'medium': 2.0, 'far': 1.0}
            distance_score = distance_scores.get(distance, 1.0)
            
            # Get danger level
            danger_score = self.get_danger_level(detection.class_name)
            
            # Get position bonus (center objects are more important)
            position = self.calculate_position(detection)
            position_scores = {'center': 1.2, 'left': 1.0, 'right': 1.0}
            position_score = position_scores.get(position, 1.0)
            
            # Calculate combined priority score
            # Higher score = higher priority
            priority = (distance_score * danger_score * position_score) + (detection.confidence * 0.5)
            
            return priority
        
        # Sort by priority score in descending order (highest priority first)
        sorted_detections = sorted(detections, key=priority_score, reverse=True)
        
        return sorted_detections
    
    def generate_spatial_description(self, detection: Detection, frame_width: int = None, frame_height: int = None) -> str:
        """
        Generate spatial description for the detection.
        
        Args:
            detection: Detection object
            frame_width: Width of the frame (optional)
            frame_height: Height of the frame (optional)
            
        Returns:
            str: Spatial description string
        """
        position = self.calculate_position(detection, frame_width)
        distance = self.estimate_distance(detection, frame_width, frame_height)
        
        # Generate natural language description
        position_phrases = {
            'left': 'to your left',
            'center': 'ahead of you',
            'right': 'to your right'
        }
        
        distance_phrases = {
            'close': 'very close',
            'medium': 'at medium distance',
            'far': 'far away'
        }
        
        position_text = position_phrases.get(position, position)
        distance_text = distance_phrases.get(distance, distance)
        
        # Create description with object name, position, and distance
        description = f"{detection.class_name} {position_text}, {distance_text}"
        
        return description
    
    def get_navigation_guidance(self, detections: List[Detection]) -> str:
        """
        Generate navigation guidance based on detected objects.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            str: Navigation guidance message
        """
        if not detections:
            return "Path appears clear"
        
        # Prioritize objects
        prioritized = self.prioritize_objects(detections)
        
        # Focus on the highest priority object
        primary_object = prioritized[0]
        position = self.calculate_position(primary_object)
        distance = self.estimate_distance(primary_object)
        danger_level = self.get_danger_level(primary_object.class_name)
        
        # Generate guidance based on position, distance, and danger
        if distance == 'close' and danger_level >= 3:
            if position == 'center':
                return f"Stop! {primary_object.class_name} directly ahead, very close"
            else:
                return f"Caution! {primary_object.class_name} {position}, very close"
        elif distance == 'close':
            if position == 'center':
                return f"Move carefully, {primary_object.class_name} ahead"
            else:
                return f"{primary_object.class_name} {position}, move cautiously"
        elif distance == 'medium' and position == 'center':
            return f"{primary_object.class_name} ahead at medium distance"
        else:
            return f"{primary_object.class_name} detected {position}"
    
    def create_navigation_contexts(self, detections: List[Detection]) -> List[NavigationContext]:
        """
        Create navigation contexts for all detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List[NavigationContext]: Navigation contexts for all detections
        """
        contexts = []
        
        for detection in detections:
            position = self.calculate_position(detection)
            distance = self.estimate_distance(detection)
            context = self.navigation_system.create_navigation_context(detection, position, distance)
            contexts.append(context)
        
        return contexts
    
    def get_contextual_navigation_guidance(self, detections: List[Detection]) -> List[str]:
        """
        Get comprehensive contextual navigation guidance for detected objects.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List[str]: List of guidance messages in priority order
        """
        # Create navigation contexts
        current_contexts = self.create_navigation_contexts(detections)
        
        # Generate comprehensive guidance
        guidance_messages = self.navigation_system.generate_comprehensive_guidance(
            current_contexts, self.previous_contexts
        )
        
        # Store contexts for next frame
        self.previous_contexts = current_contexts
        
        return guidance_messages
    
    def get_object_specific_guidance(self, detection: Detection) -> Dict[str, str]:
        """
        Get object-specific guidance for a single detection.
        
        Args:
            detection: Detection object
            
        Returns:
            Dict[str, str]: Dictionary with guidance information
        """
        position = self.calculate_position(detection)
        distance = self.estimate_distance(detection)
        context = self.navigation_system.create_navigation_context(detection, position, distance)
        
        return {
            'object': detection.class_name,
            'position': position,
            'distance': distance,
            'category': context.category.value,
            'warning_level': context.warning_level.name,
            'guidance_message': context.guidance_message,
            'action_suggestion': context.action_suggestion
        }
    
    def check_for_escalating_warnings(self, detections: List[Detection]) -> Optional[str]:
        """
        Check for escalating warnings based on object movement.
        
        Args:
            detections: Current detections
            
        Returns:
            Optional[str]: Escalating warning message if any
        """
        current_contexts = self.create_navigation_contexts(detections)
        
        if self.previous_contexts:
            return self.navigation_system.generate_escalating_warning(
                current_contexts, self.previous_contexts
            )
        
        return None
    
    def is_path_clear(self, detections: List[Detection]) -> bool:
        """
        Check if the path ahead is clear of obstacles.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            bool: True if path is clear, False otherwise
        """
        for detection in detections:
            position = self.calculate_position(detection)
            distance = self.estimate_distance(detection)
            
            # Path is blocked if there are center objects that are close or medium distance
            if position == 'center' and distance in ['close', 'medium']:
                return False
        
        return True
    
    def get_safety_priority_objects(self, detections: List[Detection]) -> List[Detection]:
        """
        Get objects that require immediate safety attention.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List[Detection]: Objects requiring immediate attention, sorted by priority
        """
        contexts = self.create_navigation_contexts(detections)
        prioritized_contexts = self.navigation_system.prioritize_contexts(contexts)
        
        # Return detections from high-priority contexts (warning level 3 or higher)
        safety_priority = [
            ctx.detection for ctx in prioritized_contexts 
            if ctx.warning_level.value >= 3
        ]
        
        return safety_priority
    
    def update_frame_dimensions(self, width: int, height: int):
        """
        Update frame dimensions for calculations.
        
        Args:
            width: New frame width
            height: New frame height
        """
        self.frame_width = width
        self.frame_height = height