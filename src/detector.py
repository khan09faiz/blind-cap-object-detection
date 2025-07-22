"""
Object detection module using YOLOv8 for enhanced blind detection system.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import logging
import torch
from ultralytics import YOLO

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Data structure for object detection results."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_point: Tuple[int, int]


class ObjectDetector:
    """YOLOv8-based object detector with GPU acceleration support."""
    
    def __init__(self, target_classes: Optional[List[str]] = None):
        """
        Initialize the object detector.
        
        Args:
            target_classes: List of class names to detect. If None, detects all classes.
        """
        self.model = None
        self.device = None
        self.target_classes = target_classes or []
        self.class_names = {}  # Will be populated when model is loaded
    
    def load_model(self, model_name: str = 'yolov8n.pt', device: str = 'auto') -> bool:
        """
        Load YOLOv8 model with specified device.
        
        Args:
            model_name: Name of the YOLO model to load (e.g., 'yolov8n.pt', 'yolov8s.pt')
            device: Device to use ('auto', 'cpu', 'cuda', or specific GPU like 'cuda:0')
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Determine device
            if device == 'auto':
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    logger.info("CUDA available, using GPU for inference")
                else:
                    self.device = 'cpu'
                    logger.info("CUDA not available, using CPU for inference")
            else:
                self.device = device
                logger.info(f"Using specified device: {device}")
            
            # Load the model
            logger.info(f"Loading YOLOv8 model: {model_name}")
            self.model = YOLO(model_name)
            
            # Move model to specified device
            if self.device != 'cpu':
                self.model.to(self.device)
            
            # Get class names from the model
            self.class_names = self.model.names
            logger.info(f"Model loaded successfully with {len(self.class_names)} classes")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def detect_objects(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Detection]:
        """
        Detect objects in the given frame.
        
        Args:
            frame: Input image frame as numpy array
            confidence_threshold: Minimum confidence threshold for detections
            
        Returns:
            List[Detection]: List of detected objects
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return []
        
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box information
                        xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                        conf = float(box.conf[0].cpu().numpy().item())  # Confidence
                        cls = int(box.cls[0].cpu().numpy().item())  # Class index
                        
                        # Skip if confidence is below threshold
                        if conf < confidence_threshold:
                            continue
                        
                        # Get class name
                        class_name = self.class_names.get(cls, f"class_{cls}")
                        
                        # Skip if not in target classes (if specified)
                        if self.target_classes and class_name not in self.target_classes:
                            continue
                        
                        # Calculate center point
                        x1, y1, x2, y2 = map(int, xyxy)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Create detection object
                        detection = Detection(
                            class_name=class_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            center_point=(center_x, center_y)
                        )
                        
                        detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            return []
    
    def filter_detections(self, detections: List[Detection], confidence_threshold: float) -> List[Detection]:
        """
        Filter detections based on confidence threshold.
        
        Args:
            detections: List of Detection objects
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List[Detection]: Filtered list of detections
        """
        filtered = [det for det in detections if det.confidence >= confidence_threshold]
        logger.debug(f"Filtered {len(detections)} detections to {len(filtered)} above threshold {confidence_threshold}")
        return filtered
    
    def filter_by_classes(self, detections: List[Detection], target_classes: List[str]) -> List[Detection]:
        """
        Filter detections to only include specified classes.
        
        Args:
            detections: List of Detection objects
            target_classes: List of class names to keep
            
        Returns:
            List[Detection]: Filtered list of detections
        """
        filtered = [det for det in detections if det.class_name in target_classes]
        logger.debug(f"Filtered {len(detections)} detections to {len(filtered)} matching target classes")
        return filtered
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including device, class names, etc.
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "status": "Model loaded",
            "device": self.device,
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names.values()),
            "target_classes": self.target_classes
        }