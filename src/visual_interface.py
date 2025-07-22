"""
Visual interface module for displaying camera feed with object detection overlay.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

from .detector import Detection
from .spatial import SpatialAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class VisualSettings:
    """Configuration for visual display settings."""
    window_name: str = "Enhanced Blind Detection System"
    window_width: int = 800
    window_height: int = 600
    show_fps: bool = True
    show_detection_count: bool = True
    show_zones: bool = True
    font_scale: float = 0.7
    font_thickness: int = 2


class VisualInterface:
    """Visual interface for displaying camera feed with detection overlay."""
    
    def __init__(self, settings: VisualSettings = None):
        """
        Initialize the visual interface.
        
        Args:
            settings: Visual display settings
        """
        self.settings = settings or VisualSettings()
        self.window_created = False
        self.frame_count = 0
        
        # Colors for different elements (BGR format)
        self.colors = {
            'person': (0, 255, 0),      # Green for people
            'vehicle': (0, 0, 255),     # Red for vehicles
            'furniture': (255, 0, 0),   # Blue for furniture
            'electronics': (255, 255, 0), # Cyan for electronics
            'default': (0, 255, 255),   # Yellow for other objects
            'zone_line': (128, 128, 128), # Gray for zone lines
            'text_bg': (0, 0, 0),       # Black for text background
            'text': (255, 255, 255),    # White for text
            'close': (0, 0, 255),       # Red for close objects
            'medium': (0, 165, 255),    # Orange for medium distance
            'far': (0, 255, 0)          # Green for far objects
        }
        
        # Object categories for color coding
        self.object_categories = {
            'person': ['person'],
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
            'furniture': ['chair', 'couch', 'bed', 'dining table', 'toilet'],
            'electronics': ['tv', 'laptop', 'cell phone', 'keyboard', 'mouse']
        }
    
    def create_window(self):
        """Create the display window."""
        if not self.window_created:
            cv2.namedWindow(self.settings.window_name, cv2.WINDOW_RESIZABLE)
            cv2.resizeWindow(self.settings.window_name, 
                           self.settings.window_width, 
                           self.settings.window_height)
            self.window_created = True
            logger.info(f"Created visual window: {self.settings.window_name}")
    
    def get_object_color(self, class_name: str, distance: str) -> Tuple[int, int, int]:
        """
        Get color for object based on category and distance.
        
        Args:
            class_name: Object class name
            distance: Distance category (close/medium/far)
            
        Returns:
            Tuple[int, int, int]: BGR color tuple
        """
        # First check distance for priority coloring
        if distance == 'close':
            return self.colors['close']
        elif distance == 'medium':
            return self.colors['medium']
        elif distance == 'far':
            return self.colors['far']
        
        # Then check object category
        for category, objects in self.object_categories.items():
            if class_name in objects:
                return self.colors[category]
        
        return self.colors['default']
    
    def draw_detection_box(self, frame: np.ndarray, detection: Detection, 
                          position: str, distance: str) -> np.ndarray:
        """
        Draw detection bounding box with information.
        
        Args:
            frame: Input frame
            detection: Detection object
            position: Spatial position (left/center/right)
            distance: Distance category
            
        Returns:
            np.ndarray: Frame with detection box drawn
        """
        x1, y1, x2, y2 = map(int, detection.bbox)
        color = self.get_object_color(detection.class_name, distance)
        
        # Draw bounding box
        thickness = 3 if distance == 'close' else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        confidence_pct = int(detection.confidence * 100)
        label = f"{detection.class_name} ({confidence_pct}%)"
        distance_label = f"{distance.upper()} - {position.upper()}"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.settings.font_scale
        font_thickness = self.settings.font_thickness
        
        (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        (dist_width, dist_height), _ = cv2.getTextSize(distance_label, font, font_scale - 0.1, font_thickness - 1)
        
        # Draw text background
        text_bg_height = label_height + dist_height + 10
        text_bg_width = max(label_width, dist_width) + 10
        
        cv2.rectangle(frame, 
                     (x1, y1 - text_bg_height - 5), 
                     (x1 + text_bg_width, y1), 
                     self.colors['text_bg'], -1)
        
        # Draw text
        cv2.putText(frame, label, (x1 + 5, y1 - dist_height - 10), 
                   font, font_scale, color, font_thickness)
        cv2.putText(frame, distance_label, (x1 + 5, y1 - 5), 
                   font, font_scale - 0.1, color, font_thickness - 1)
        
        return frame
    
    def draw_zone_lines(self, frame: np.ndarray, spatial_analyzer: SpatialAnalyzer) -> np.ndarray:
        """
        Draw zone boundary lines.
        
        Args:
            frame: Input frame
            spatial_analyzer: Spatial analyzer instance
            
        Returns:
            np.ndarray: Frame with zone lines drawn
        """
        if not self.settings.show_zones:
            return frame
        
        height, width = frame.shape[:2]
        
        # Calculate zone boundaries (assuming 1/3 splits)
        left_boundary = width // 3
        right_boundary = 2 * width // 3
        
        # Draw vertical zone lines
        cv2.line(frame, (left_boundary, 0), (left_boundary, height), 
                self.colors['zone_line'], 1, cv2.LINE_AA)
        cv2.line(frame, (right_boundary, 0), (right_boundary, height), 
                self.colors['zone_line'], 1, cv2.LINE_AA)
        
        # Add zone labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "LEFT", (10, 30), font, 0.6, self.colors['zone_line'], 2)
        cv2.putText(frame, "CENTER", (left_boundary + 10, 30), font, 0.6, self.colors['zone_line'], 2)
        cv2.putText(frame, "RIGHT", (right_boundary + 10, 30), font, 0.6, self.colors['zone_line'], 2)
        
        return frame
    
    def draw_info_panel(self, frame: np.ndarray, fps: float, 
                       detection_count: int, detections: List[Tuple[Detection, str, str]]) -> np.ndarray:
        """
        Draw information panel with system stats.
        
        Args:
            frame: Input frame
            fps: Current FPS
            detection_count: Number of detections
            detections: List of current detections
            
        Returns:
            np.ndarray: Frame with info panel drawn
        """
        height, width = frame.shape[:2]
        
        # Info panel background
        panel_height = 120
        cv2.rectangle(frame, (0, height - panel_height), (width, height), 
                     (0, 0, 0, 180), -1)  # Semi-transparent black
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = self.colors['text']
        
        y_offset = height - panel_height + 25
        
        # FPS and detection count
        if self.settings.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), 
                       font, font_scale, text_color, font_thickness)
        
        if self.settings.show_detection_count:
            cv2.putText(frame, f"Objects: {detection_count}", (150, y_offset), 
                       font, font_scale, text_color, font_thickness)
        
        # Current detections summary
        y_offset += 25
        close_objects = [d for d, p, dist in detections if dist == 'close']
        center_objects = [d for d, p, dist in detections if p == 'center']
        
        if close_objects:
            cv2.putText(frame, f"CLOSE: {len(close_objects)} objects", (10, y_offset), 
                       font, font_scale, self.colors['close'], font_thickness)
        
        if center_objects:
            cv2.putText(frame, f"PATH: {len(center_objects)} objects", (200, y_offset), 
                       font, font_scale, self.colors['medium'], font_thickness)
        
        # Distance legend
        y_offset += 25
        cv2.putText(frame, "Close", (10, y_offset), font, 0.5, self.colors['close'], 1)
        cv2.putText(frame, "Medium", (80, y_offset), font, 0.5, self.colors['medium'], 1)
        cv2.putText(frame, "Far", (150, y_offset), font, 0.5, self.colors['far'], 1)
        
        # Instructions
        y_offset += 20
        cv2.putText(frame, "Press 'q' to quit, 's' to screenshot", (10, y_offset), 
                   font, 0.4, text_color, 1)
        
        return frame
    
    def draw_crosshair(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw center crosshair for reference.
        
        Args:
            frame: Input frame
            
        Returns:
            np.ndarray: Frame with crosshair drawn
        """
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Draw crosshair
        cross_size = 20
        color = self.colors['zone_line']
        
        cv2.line(frame, (center_x - cross_size, center_y), 
                (center_x + cross_size, center_y), color, 2)
        cv2.line(frame, (center_x, center_y - cross_size), 
                (center_x, center_y + cross_size), color, 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, detections: List[Tuple[Detection, str, str]], 
                     spatial_analyzer: SpatialAnalyzer, fps: float = 0.0) -> np.ndarray:
        """
        Process frame with all visual overlays.
        
        Args:
            frame: Input frame
            detections: List of (Detection, position, distance) tuples
            spatial_analyzer: Spatial analyzer instance
            fps: Current FPS
            
        Returns:
            np.ndarray: Processed frame with overlays
        """
        # Make a copy to avoid modifying original
        display_frame = frame.copy()
        
        # Draw zone lines
        display_frame = self.draw_zone_lines(display_frame, spatial_analyzer)
        
        # Draw crosshair
        display_frame = self.draw_crosshair(display_frame)
        
        # Draw detections
        for detection, position, distance in detections:
            display_frame = self.draw_detection_box(display_frame, detection, position, distance)
        
        # Draw info panel
        display_frame = self.draw_info_panel(display_frame, fps, len(detections), detections)
        
        self.frame_count += 1
        return display_frame
    
    def show_frame(self, frame: np.ndarray) -> bool:
        """
        Display the frame in the window.
        
        Args:
            frame: Frame to display
            
        Returns:
            bool: True if should continue, False if should exit
        """
        if not self.window_created:
            self.create_window()
        
        cv2.imshow(self.settings.window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False  # Quit
        elif key == ord('s'):
            # Save screenshot
            filename = f"detection_screenshot_{self.frame_count}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Screenshot saved: {filename}")
        elif key == ord('z'):
            # Toggle zone lines
            self.settings.show_zones = not self.settings.show_zones
        elif key == ord('f'):
            # Toggle FPS display
            self.settings.show_fps = not self.settings.show_fps
        
        return True
    
    def cleanup(self):
        """Clean up visual interface resources."""
        if self.window_created:
            cv2.destroyWindow(self.settings.window_name)
            self.window_created = False
            logger.info("Visual interface cleaned up")
    
    def resize_frame(self, frame: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
        """
        Resize frame to fit display window while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            np.ndarray: Resized frame
        """
        height, width = frame.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def add_detection_statistics(self, frame: np.ndarray, 
                               detection_history: List[int]) -> np.ndarray:
        """
        Add detection statistics overlay.
        
        Args:
            frame: Input frame
            detection_history: List of recent detection counts
            
        Returns:
            np.ndarray: Frame with statistics overlay
        """
        if not detection_history:
            return frame
        
        height, width = frame.shape[:2]
        
        # Simple graph of detection counts
        graph_width = 200
        graph_height = 50
        graph_x = width - graph_width - 10
        graph_y = 10
        
        # Draw graph background
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), 
                     (0, 0, 0, 128), -1)
        
        # Draw detection count line
        if len(detection_history) > 1:
            max_detections = max(detection_history) if detection_history else 1
            points = []
            
            for i, count in enumerate(detection_history[-graph_width:]):
                x = graph_x + i
                y = graph_y + graph_height - int((count / max_detections) * graph_height)
                points.append((x, y))
            
            # Draw line
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 255, 0), 1)
        
        # Add label
        cv2.putText(frame, f"Max: {max(detection_history)}", 
                   (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 255), 1)
        
        return frame


def create_visual_interface(enable_visual: bool = True, 
                          window_size: Tuple[int, int] = (800, 600)) -> Optional[VisualInterface]:
    """
    Create and configure visual interface.
    
    Args:
        enable_visual: Whether to enable visual display
        window_size: Window size (width, height)
        
    Returns:
        Optional[VisualInterface]: Visual interface instance or None
    """
    if not enable_visual:
        return None
    
    settings = VisualSettings(
        window_width=window_size[0],
        window_height=window_size[1],
        show_fps=True,
        show_detection_count=True,
        show_zones=True
    )
    
    return VisualInterface(settings)
