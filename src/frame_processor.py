"""
Frame processing module for camera handling and frame preprocessing.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from .config import Config


class CameraError(Exception):
    """Custom exception for camera-related errors."""
    pass


class CameraConnectionError(CameraError):
    """Exception for camera connection issues."""
    pass


class CameraConfigurationError(CameraError):
    """Exception for camera configuration issues."""
    pass


class FrameProcessor:
    """Handle video capture and frame preprocessing with enhanced error handling."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize FrameProcessor with optional configuration.
        
        Args:
            config: Configuration object with camera settings
        """
        self.camera = None
        self.is_initialized = False
        self.config = config or Config()
        self.camera_index = self.config.camera_index
        self.frame_width = self.config.frame_width
        self.frame_height = self.config.frame_height
        self.logger = logging.getLogger(__name__)
        self._retry_count = 0
        self._max_retries = 3
    
    def initialize_camera(self, camera_index: Optional[int] = None, 
                         frame_width: Optional[int] = None, 
                         frame_height: Optional[int] = None) -> bool:
        """
        Initialize camera with specified settings or config defaults.
        
        Args:
            camera_index: Camera device index (uses config default if None)
            frame_width: Desired frame width (uses config default if None)
            frame_height: Desired frame height (uses config default if None)
            
        Returns:
            bool: True if camera initialized successfully, False otherwise
            
        Raises:
            CameraConnectionError: If camera cannot be connected
            CameraConfigurationError: If camera settings cannot be applied
        """
        # Use provided values or fall back to config
        self.camera_index = camera_index if camera_index is not None else self.config.camera_index
        self.frame_width = frame_width if frame_width is not None else self.config.frame_width
        self.frame_height = frame_height if frame_height is not None else self.config.frame_height
        
        # Release any existing camera
        if self.camera is not None:
            self.release_camera()
        
        # Try to initialize camera with retry logic
        for attempt in range(self._max_retries):
            try:
                self.logger.info(f"Attempting to initialize camera {self.camera_index} (attempt {attempt + 1}/{self._max_retries})")
                
                # Check if camera is available first
                if not self.check_camera_availability(self.camera_index):
                    available_cameras = self.list_available_cameras()
                    if available_cameras:
                        self.logger.warning(f"Camera {self.camera_index} not available. Available cameras: {available_cameras}")
                        if attempt == 0:  # Try first available camera on first retry
                            self.camera_index = available_cameras[0]['index']
                            self.logger.info(f"Switching to camera {self.camera_index}")
                        else:
                            raise CameraConnectionError(f"Camera {self.camera_index} not available. Available: {available_cameras}")
                    else:
                        raise CameraConnectionError("No cameras available on the system")
                
                # Initialize camera
                self.camera = cv2.VideoCapture(self.camera_index)
                
                if not self.camera.isOpened():
                    raise CameraConnectionError(f"Cannot open camera at index {self.camera_index}")
                
                # Configure camera properties
                self._configure_camera_properties()
                
                # Verify camera is working
                self._verify_camera_functionality()
                
                self.is_initialized = True
                self._retry_count = 0
                self.logger.info(f"Camera {self.camera_index} initialized successfully")
                return True
                
            except (CameraConnectionError, CameraConfigurationError) as e:
                self.logger.warning(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
                
                if attempt == self._max_retries - 1:
                    self.is_initialized = False
                    raise e
                
                # Wait before retry (exponential backoff)
                import time
                time.sleep(0.5 * (2 ** attempt))
            
            except Exception as e:
                self.logger.error(f"Unexpected error during camera initialization: {e}")
                self.is_initialized = False
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
                raise CameraError(f"Camera initialization failed: {e}")
        
        return False
    
    def _configure_camera_properties(self) -> None:
        """Configure camera properties and validate settings."""
        try:
            # Set basic properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Set additional properties for better performance
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            
            # Verify settings were applied
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            # Warn if settings don't match requested values
            if actual_width != self.frame_width or actual_height != self.frame_height:
                self.logger.warning(f"Camera resolution mismatch: requested {self.frame_width}x{self.frame_height}, got {actual_width}x{actual_height}")
                # Update our internal values to match actual camera settings
                self.frame_width = actual_width
                self.frame_height = actual_height
                
        except Exception as e:
            raise CameraConfigurationError(f"Failed to configure camera properties: {e}")
    
    def _verify_camera_functionality(self) -> None:
        """Verify camera can capture frames properly."""
        try:
            # Test frame capture multiple times to ensure stability
            for i in range(3):
                ret, test_frame = self.camera.read()
                if not ret or test_frame is None:
                    raise CameraConnectionError(f"Camera test capture {i+1} failed")
                
                # Verify frame properties
                if len(test_frame.shape) != 3 or test_frame.shape[2] != 3:
                    raise CameraConnectionError(f"Invalid frame format: {test_frame.shape}")
                
                # Small delay between test captures
                import time
                time.sleep(0.1)
                
            self.logger.debug("Camera functionality verification passed")
            
        except Exception as e:
            raise CameraConnectionError(f"Camera functionality verification failed: {e}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera with automatic recovery.
        
        Returns:
            Optional[np.ndarray]: Captured frame or None if capture fails
            
        Raises:
            CameraError: If camera is not initialized or frame capture fails persistently
        """
        if not self.is_initialized or self.camera is None:
            raise CameraError("Camera not initialized. Call initialize_camera() first.")
        
        try:
            # Check if camera is still connected
            if not self.camera.isOpened():
                self.logger.warning("Camera connection lost, attempting to reconnect...")
                self._attempt_camera_recovery()
            
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                self.logger.warning("Failed to capture frame from camera")
                self._retry_count += 1
                
                # Attempt recovery if multiple failures
                if self._retry_count >= 3:
                    self.logger.warning("Multiple frame capture failures, attempting camera recovery...")
                    self._attempt_camera_recovery()
                
                return None
            
            # Reset retry count on successful capture
            self._retry_count = 0
            return frame
            
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            self._retry_count += 1
            
            if self._retry_count >= 5:
                raise CameraError(f"Persistent frame capture failures: {e}")
            
            return None
    
    def _attempt_camera_recovery(self) -> None:
        """Attempt to recover camera connection."""
        try:
            self.logger.info("Attempting camera recovery...")
            
            # Release current camera
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            
            # Wait briefly before reconnection
            import time
            time.sleep(1.0)
            
            # Reinitialize camera
            self.is_initialized = False
            self.initialize_camera(self.camera_index, self.frame_width, self.frame_height)
            
            self.logger.info("Camera recovery successful")
            
        except Exception as e:
            self.logger.error(f"Camera recovery failed: {e}")
            self.is_initialized = False
            raise CameraError(f"Camera recovery failed: {e}")
    
    def preprocess_frame(self, frame: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """
        Preprocess frame for detection with enhanced error handling.
        
        Args:
            frame: Input frame from camera
            enhance_contrast: Whether to apply contrast enhancement
            
        Returns:
            np.ndarray: Preprocessed frame ready for detection
            
        Raises:
            ValueError: If frame is invalid
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: frame is None or empty")
        
        try:
            # Validate frame format
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError(f"Invalid frame shape: {frame.shape}. Expected (H, W, 3)")
            
            # Check for reasonable frame dimensions
            height, width = frame.shape[:2]
            if width < 32 or height < 32:
                raise ValueError(f"Frame too small: {width}x{height}. Minimum size is 32x32")
            
            if width > 4096 or height > 4096:
                self.logger.warning(f"Very large frame: {width}x{height}. This may impact performance")
            
            # Resize frame if it doesn't match expected dimensions
            if width != self.frame_width or height != self.frame_height:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height), 
                                 interpolation=cv2.INTER_LINEAR)
                self.logger.debug(f"Resized frame from {width}x{height} to {self.frame_width}x{self.frame_height}")
            
            # Convert to RGB for YOLO (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply contrast enhancement if requested
            if enhance_contrast:
                try:
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    enhanced_frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                    
                    # Verify enhancement didn't corrupt the frame
                    if enhanced_frame_rgb.shape == frame_rgb.shape:
                        frame_rgb = enhanced_frame_rgb
                    else:
                        self.logger.warning("Contrast enhancement corrupted frame, using original")
                        
                except Exception as e:
                    self.logger.warning(f"Contrast enhancement failed: {e}, using original frame")
            
            # Final validation
            if frame_rgb.shape != (self.frame_height, self.frame_width, 3):
                raise ValueError(f"Preprocessed frame has wrong shape: {frame_rgb.shape}")
            
            return frame_rgb
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            self.logger.error(f"Error preprocessing frame: {e}")
            # Attempt basic fallback processing
            try:
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Just convert to RGB without enhancement
                    fallback_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if fallback_frame.shape[:2] != (self.frame_height, self.frame_width):
                        fallback_frame = cv2.resize(fallback_frame, (self.frame_width, self.frame_height))
                    return fallback_frame
                else:
                    raise ValueError("Cannot recover from preprocessing error")
            except Exception as fallback_error:
                raise ValueError(f"Frame preprocessing failed completely: {e}, fallback failed: {fallback_error}")
    
    def get_camera_info(self) -> dict:
        """
        Get current camera information and settings.
        
        Returns:
            dict: Camera information including resolution, FPS, etc.
        """
        if not self.is_initialized or self.camera is None:
            return {"status": "not_initialized"}
        
        try:
            info = {
                "status": "initialized",
                "camera_index": self.camera_index,
                "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.camera.get(cv2.CAP_PROP_FPS),
                "backend": self.camera.getBackendName(),
                "is_opened": self.camera.isOpened()
            }
            return info
        except Exception as e:
            self.logger.error(f"Error getting camera info: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_camera_availability(self, camera_index: int = 0) -> bool:
        """
        Check if a camera is available at the given index.
        
        Args:
            camera_index: Camera device index to check
            
        Returns:
            bool: True if camera is available, False otherwise
        """
        try:
            test_camera = cv2.VideoCapture(camera_index)
            is_available = test_camera.isOpened()
            
            # Additional check: try to read a frame
            if is_available:
                ret, frame = test_camera.read()
                is_available = ret and frame is not None
            
            test_camera.release()
            return is_available
        except Exception as e:
            self.logger.debug(f"Camera {camera_index} availability check failed: {e}")
            return False
    
    def list_available_cameras(self, max_cameras: int = 10) -> List[dict]:
        """
        List all available cameras with detailed information.
        
        Args:
            max_cameras: Maximum number of camera indices to check
            
        Returns:
            List[dict]: List of available camera information
        """
        available_cameras = []
        
        for i in range(max_cameras):
            if self.check_camera_availability(i):
                try:
                    # Get camera details
                    test_camera = cv2.VideoCapture(i)
                    if test_camera.isOpened():
                        camera_info = {
                            'index': i,
                            'width': int(test_camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            'height': int(test_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            'fps': test_camera.get(cv2.CAP_PROP_FPS),
                            'backend': test_camera.getBackendName()
                        }
                        available_cameras.append(camera_info)
                    test_camera.release()
                except Exception as e:
                    self.logger.debug(f"Error getting details for camera {i}: {e}")
                    # Add basic info even if details fail
                    available_cameras.append({'index': i, 'error': str(e)})
        
        return available_cameras
    
    def release_camera(self) -> None:
        """Release camera resources and cleanup."""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                self.logger.info("Camera resources released")
            
            self.is_initialized = False
            
        except Exception as e:
            self.logger.error(f"Error releasing camera: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure camera is released."""
        self.release_camera()
    
    def get_frame_rate(self) -> float:
        """
        Get current camera frame rate.
        
        Returns:
            float: Current FPS or 0.0 if camera not initialized
        """
        if not self.is_initialized or self.camera is None:
            return 0.0
        
        try:
            return self.camera.get(cv2.CAP_PROP_FPS)
        except Exception:
            return 0.0
    
    def set_frame_rate(self, fps: float) -> bool:
        """
        Set camera frame rate.
        
        Args:
            fps: Desired frames per second
            
        Returns:
            bool: True if successfully set, False otherwise
        """
        if not self.is_initialized or self.camera is None:
            return False
        
        try:
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"Frame rate set to {actual_fps} FPS (requested: {fps})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set frame rate: {e}")
            return False
    
    def capture_frame_with_timeout(self, timeout_seconds: float = 5.0) -> Optional[np.ndarray]:
        """
        Capture frame with timeout to prevent hanging.
        
        Args:
            timeout_seconds: Maximum time to wait for frame
            
        Returns:
            Optional[np.ndarray]: Captured frame or None if timeout/failure
        """
        if not self.is_initialized or self.camera is None:
            raise CameraError("Camera not initialized. Call initialize_camera() first.")
        
        import threading
        import time
        
        frame_result = [None]
        exception_result = [None]
        
        def capture_worker():
            try:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    frame_result[0] = frame
            except Exception as e:
                exception_result[0] = e
        
        # Start capture in separate thread
        capture_thread = threading.Thread(target=capture_worker)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Wait for completion or timeout
        capture_thread.join(timeout=timeout_seconds)
        
        if capture_thread.is_alive():
            self.logger.warning(f"Frame capture timed out after {timeout_seconds} seconds")
            return None
        
        if exception_result[0]:
            self.logger.error(f"Frame capture error: {exception_result[0]}")
            return None
        
        return frame_result[0]
    
    def get_camera_properties(self) -> dict:
        """
        Get comprehensive camera properties.
        
        Returns:
            dict: Dictionary of camera properties
        """
        if not self.is_initialized or self.camera is None:
            return {"status": "not_initialized"}
        
        try:
            properties = {
                "status": "initialized",
                "camera_index": self.camera_index,
                "frame_width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "frame_height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.camera.get(cv2.CAP_PROP_FPS),
                "backend": self.camera.getBackendName(),
                "is_opened": self.camera.isOpened(),
                "brightness": self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.camera.get(cv2.CAP_PROP_CONTRAST),
                "saturation": self.camera.get(cv2.CAP_PROP_SATURATION),
                "hue": self.camera.get(cv2.CAP_PROP_HUE),
                "gain": self.camera.get(cv2.CAP_PROP_GAIN),
                "exposure": self.camera.get(cv2.CAP_PROP_EXPOSURE),
                "buffer_size": self.camera.get(cv2.CAP_PROP_BUFFERSIZE)
            }
            return properties
        except Exception as e:
            self.logger.error(f"Error getting camera properties: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_camera_performance(self, num_frames: int = 30) -> dict:
        """
        Test camera performance by capturing multiple frames.
        
        Args:
            num_frames: Number of frames to capture for testing
            
        Returns:
            dict: Performance metrics
        """
        if not self.is_initialized or self.camera is None:
            return {"error": "Camera not initialized"}
        
        import time
        
        start_time = time.time()
        successful_captures = 0
        failed_captures = 0
        frame_times = []
        
        for i in range(num_frames):
            frame_start = time.time()
            
            try:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    successful_captures += 1
                    frame_time = time.time() - frame_start
                    frame_times.append(frame_time)
                else:
                    failed_captures += 1
            except Exception:
                failed_captures += 1
        
        total_time = time.time() - start_time
        
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            avg_frame_time = 0
            actual_fps = 0
        
        return {
            "total_frames": num_frames,
            "successful_captures": successful_captures,
            "failed_captures": failed_captures,
            "success_rate": successful_captures / num_frames if num_frames > 0 else 0,
            "total_time": total_time,
            "average_frame_time": avg_frame_time,
            "actual_fps": actual_fps,
            "theoretical_fps": self.get_frame_rate()
        }