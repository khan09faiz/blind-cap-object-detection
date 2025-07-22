"""
Main application controller for the enhanced blind detection system.
"""

import signal
import sys
import time
import threading
from typing import Optional, List, Tuple

from .config import ConfigManager
from .detector import ObjectDetector, Detection
from .spatial import SpatialAnalyzer
from .audio import AudioManager
from .frame_processor import FrameProcessor, CameraError
from .visual_interface import VisualInterface, create_visual_interface
from .logging_config import initialize_logging, get_logger, shutdown_logging, LoggingContextManager
from .error_handling import (
    ErrorHandler, SystemError, CameraError as CameraErrorCustom, 
    DetectionError, AudioError, ErrorSeverity, HealthChecker,
    error_handler, ErrorCategory, safe_execute
)
from .performance import PerformanceMonitor, PerformanceOptimizer, initialize_performance_system


class BlindDetectionApp:
    """Main application class that orchestrates all components."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the main application controller.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize logging system first
        self.logging_manager = initialize_logging()
        self.logger = get_logger(__name__)
        
        # Initialize error handling
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize health checker
        self.health_checker = HealthChecker(self.logger)
        
        try:
            with LoggingContextManager(self.logger, "application initialization"):
                # Load configuration
                self.config_manager = ConfigManager(config_path)
                self.config = self.config_manager.load_config()
                
                # Initialize performance monitoring system
                self.performance_monitor, self.performance_optimizer = initialize_performance_system(self.config)
                self.logger.info("Performance monitoring system initialized")
                
                # Initialize components with configuration
                self.detector = None
                self.spatial_analyzer = None
                self.audio_manager = None
                self.frame_processor = None
                
                # Application state
                self.running = False
                self.initialized = False
                self.last_path_clear_state = None
                
                # Performance tracking
                self.frame_count = 0
                self.start_time = None
                self.last_fps_report = 0
                self.performance_logger = self.logging_manager.create_performance_logger()
                
                # Setup signal handlers for graceful shutdown
                signal.signal(signal.SIGINT, self.signal_handler)
                signal.signal(signal.SIGTERM, self.signal_handler)
                
                # Register health checks
                self._register_health_checks()
                
                self.logger.info("BlindDetectionApp initialized successfully")
                
        except Exception as e:
            error = SystemError(f"Failed to initialize application: {e}", 
                              ErrorSeverity.CRITICAL,
                              "Check configuration file and system dependencies")
            self.error_handler.handle_error(error, "application initialization")
            raise
    
    def _register_health_checks(self) -> None:
        """Register system health checks."""
        try:
            # Camera health check
            def camera_health():
                return (self.frame_processor is not None and 
                       self.frame_processor.is_initialized and
                       self.frame_processor.camera is not None and
                       self.frame_processor.camera.isOpened())
            
            # Detector health check
            def detector_health():
                return (self.detector is not None and 
                       self.detector.model is not None)
            
            # Audio health check
            def audio_health():
                return (self.audio_manager is not None and 
                       self.audio_manager.tts_engine is not None)
            
            # System health check
            def system_health():
                return self.initialized and self.running
            
            self.health_checker.register_health_check("camera", camera_health)
            self.health_checker.register_health_check("detector", detector_health)
            self.health_checker.register_health_check("audio", audio_health)
            self.health_checker.register_health_check("system", system_health)
            
            self.logger.debug("Health checks registered successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to register health checks: {e}")
    
    @error_handler(ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL, 
                  "Check system dependencies and configuration")
    def initialize(self) -> bool:
        """Initialize all system components with comprehensive error handling."""
        with LoggingContextManager(self.logger, "system initialization"):
            try:
                # Validate configuration first
                if not safe_execute(self.config_manager.validate_config, 
                                  default_return=False, logger=self.logger):
                    raise SystemError("Configuration validation failed",
                                    ErrorSeverity.CRITICAL,
                                    "Check config.yaml file format and values")
                
                # Initialize frame processor with camera
                self.logger.info("Initializing camera and frame processor...")
                self.frame_processor = FrameProcessor(self.config)
                
                try:
                    if not self.frame_processor.initialize_camera(
                        self.config.camera_index,
                        self.config.frame_width,
                        self.config.frame_height
                    ):
                        raise CameraErrorCustom("Failed to initialize camera",
                                               ErrorSeverity.CRITICAL,
                                               "Check camera connection and permissions")
                except Exception as e:
                    camera_error = CameraErrorCustom(f"Camera initialization error: {e}",
                                                   ErrorSeverity.CRITICAL,
                                                   "Verify camera is connected and not in use by another application")
                    self.error_handler.handle_error(camera_error, "camera initialization")
                    return False
                
                # Initialize object detector
                self.logger.info("Initializing object detector...")
                self.detector = ObjectDetector(target_classes=self.config.target_classes)
                
                try:
                    if not self.detector.load_model(self.config.model_name, self.config.device):
                        raise DetectionError("Failed to load detection model",
                                           ErrorSeverity.CRITICAL,
                                           "Check model file exists and GPU/CUDA setup")
                except Exception as e:
                    detection_error = DetectionError(f"Model loading error: {e}",
                                                   ErrorSeverity.CRITICAL,
                                                   "Verify model file and PyTorch installation")
                    self.error_handler.handle_error(detection_error, "model loading")
                    return False
                
                # Initialize spatial analyzer
                self.logger.info("Initializing spatial analyzer...")
                try:
                    self.spatial_analyzer = SpatialAnalyzer(
                        frame_width=self.config.frame_width,
                        frame_height=self.config.frame_height
                    )
                except Exception as e:
                    spatial_error = SystemError(f"Spatial analyzer initialization failed: {e}",
                                               ErrorSeverity.HIGH,
                                               "Check frame dimensions in configuration")
                    self.error_handler.handle_error(spatial_error, "spatial analyzer initialization")
                    return False
                
                # Initialize audio manager
                self.logger.info("Initializing audio manager...")
                try:
                    self.audio_manager = AudioManager(
                        voice_rate=self.config.voice_rate,
                        voice_volume=self.config.voice_volume,
                        announcement_cooldown=self.config.announcement_cooldown
                    )
                except Exception as e:
                    audio_error = AudioError(f"Audio manager initialization failed: {e}",
                                           ErrorSeverity.MEDIUM,
                                           "Check audio system and TTS engine installation")
                    self.error_handler.handle_error(audio_error, "audio manager initialization")
                    # Continue without audio - system can still work
                    self.audio_manager = None
                
                # Verify critical components are ready
                critical_components = [self.frame_processor, self.detector, self.spatial_analyzer]
                if not all(critical_components):
                    raise SystemError("Critical components failed to initialize",
                                    ErrorSeverity.CRITICAL,
                                    "Check system logs for specific component failures")
                
                # Run initial health check
                health_results = self.health_checker.run_health_checks()
                failed_checks = [name for name, result in health_results.items() if not result]
                
                if failed_checks:
                    self.logger.warning(f"Health check failures: {failed_checks}")
                    if 'camera' in failed_checks or 'detector' in failed_checks:
                        raise SystemError(f"Critical health checks failed: {failed_checks}",
                                        ErrorSeverity.CRITICAL,
                                        "Check camera and model initialization")
                
                self.initialized = True
                self.logger.info("System initialization completed successfully")
                
                # Announce system ready if audio is available
                if self.audio_manager:
                    safe_execute(
                        lambda: self.audio_manager.announce("Enhanced blind detection system ready"),
                        logger=self.logger
                    )
                
                return True
                
            except (SystemError, CameraErrorCustom, DetectionError, AudioError) as e:
                # These are already handled by the error handler
                return False
            except Exception as e:
                # Unexpected error
                system_error = SystemError(f"Unexpected initialization error: {e}",
                                         ErrorSeverity.CRITICAL,
                                         "Check system logs and restart application")
                self.error_handler.handle_error(system_error, "system initialization")
                return False
    
    def process_frame(self) -> bool:
        """
        Process a single frame through the detection pipeline with comprehensive error handling.
        
        Returns:
            bool: True if frame processed successfully, False otherwise
        """
        with self.performance_monitor.measure_frame():
            try:
                # Capture frame from camera with error handling
                try:
                    raw_frame = self.frame_processor.capture_frame()
                    if raw_frame is None:
                        camera_error = CameraErrorCustom("Failed to capture frame from camera",
                                                        ErrorSeverity.MEDIUM,
                                                        "Check camera connection and restart if necessary")
                        self.error_handler.handle_error(camera_error, "frame capture")
                        return False
                except Exception as e:
                    camera_error = CameraErrorCustom(f"Camera capture error: {e}",
                                                   ErrorSeverity.HIGH,
                                                   "Camera may need to be reinitialized")
                    self.error_handler.handle_error(camera_error, "frame capture")
                    return False
                
                # Preprocess frame for detection with error handling
                try:
                    processed_frame = self.frame_processor.preprocess_frame(raw_frame)
                except Exception as e:
                    processing_error = SystemError(f"Frame preprocessing failed: {e}",
                                                 ErrorSeverity.MEDIUM,
                                                 "Check frame format and preprocessing parameters")
                    self.error_handler.handle_error(processing_error, "frame preprocessing")
                    return False
                
                # Detect objects in the frame with error handling
                try:
                    with self.performance_monitor.measure_detection():
                        detections = self.detector.detect_objects(
                            processed_frame, 
                            confidence_threshold=self.config.confidence_threshold
                        )
                except Exception as e:
                    detection_error = DetectionError(f"Object detection failed: {e}",
                                                   ErrorSeverity.MEDIUM,
                                                   "Check model and input frame format")
                    self.error_handler.handle_error(detection_error, "object detection")
                    return False
                
                # Process detections if any found
                if detections:
                    try:
                        with self.performance_monitor.measure_audio():
                            self.process_detections(detections)
                    except Exception as e:
                        processing_error = SystemError(f"Detection processing failed: {e}",
                                                     ErrorSeverity.MEDIUM,
                                                     "Check spatial analysis and audio systems")
                        self.error_handler.handle_error(processing_error, "detection processing")
                        # Continue processing even if this fails
                else:
                    # Check if path was previously blocked and now clear
                    current_path_clear = True
                    if self.last_path_clear_state is False and self.audio_manager:
                        with self.performance_monitor.measure_audio():
                            safe_execute(
                                lambda: self.audio_manager.announce_path_status(
                                    current_path_clear, self.last_path_clear_state
                                ),
                                logger=self.logger
                            )
                    self.last_path_clear_state = current_path_clear
                
                # Update frame counter for performance tracking
                self.frame_count += 1
                self.report_performance()
                
                return True
                
            except Exception as e:
                # Catch-all for unexpected errors
                system_error = SystemError(f"Unexpected error in frame processing: {e}",
                                         ErrorSeverity.HIGH,
                                     "Check system logs and consider restarting")
            self.error_handler.handle_error(system_error, "frame processing")
            return False
    
    def process_detections(self, detections: List[Detection]) -> None:
        """
        Process detected objects through spatial analysis and audio feedback with error handling.
        
        Args:
            detections: List of detected objects
        """
        try:
            # Prioritize objects based on spatial analysis
            try:
                prioritized_detections = self.spatial_analyzer.prioritize_objects(detections)
            except Exception as e:
                spatial_error = SystemError(f"Object prioritization failed: {e}",
                                           ErrorSeverity.MEDIUM,
                                           "Using original detection order as fallback")
                self.error_handler.handle_error(spatial_error, "object prioritization")
                prioritized_detections = detections  # Fallback to original order
            
            # Get comprehensive navigation guidance
            guidance_messages = []
            try:
                guidance_messages = self.spatial_analyzer.get_contextual_navigation_guidance(
                    prioritized_detections
                )
            except Exception as e:
                spatial_error = SystemError(f"Navigation guidance generation failed: {e}",
                                           ErrorSeverity.MEDIUM,
                                           "Basic object announcements will be used instead")
                self.error_handler.handle_error(spatial_error, "navigation guidance")
            
            # Check for escalating warnings
            escalating_warning = None
            try:
                escalating_warning = self.spatial_analyzer.check_for_escalating_warnings(
                    prioritized_detections
                )
            except Exception as e:
                spatial_error = SystemError(f"Escalating warning check failed: {e}",
                                           ErrorSeverity.LOW,
                                           "Continuing without escalating warnings")
                self.error_handler.handle_error(spatial_error, "escalating warning check")
            
            # Announce escalating warning first if present and audio available
            if escalating_warning and self.audio_manager:
                safe_execute(
                    lambda: self.audio_manager.announce_escalating_warning(escalating_warning),
                    logger=self.logger
                )
            
            # Announce navigation guidance if audio available
            if guidance_messages and self.audio_manager:
                safe_execute(
                    lambda: self.audio_manager.announce_navigation_guidance(guidance_messages),
                    logger=self.logger
                )
            
            # Update path clear state
            try:
                current_path_clear = self.spatial_analyzer.is_path_clear(prioritized_detections)
                
                if (self.last_path_clear_state is not None and 
                    self.last_path_clear_state != current_path_clear and 
                    self.audio_manager):
                    
                    safe_execute(
                        lambda: self.audio_manager.announce_path_status(
                            current_path_clear, self.last_path_clear_state
                        ),
                        logger=self.logger
                    )
                
                self.last_path_clear_state = current_path_clear
                
            except Exception as e:
                spatial_error = SystemError(f"Path clear state update failed: {e}",
                                           ErrorSeverity.LOW,
                                           "Path status announcements may be affected")
                self.error_handler.handle_error(spatial_error, "path clear state")
            
            # Log detection summary
            self.logger.debug(f"Processed {len(detections)} detections, "
                            f"prioritized: {len(prioritized_detections)}, "
                            f"guidance messages: {len(guidance_messages)}")
            
        except Exception as e:
            # Catch-all for unexpected errors in detection processing
            system_error = SystemError(f"Unexpected error processing detections: {e}",
                                     ErrorSeverity.MEDIUM,
                                     "Detection processing will continue with next frame")
            self.error_handler.handle_error(system_error, "detection processing")
            raise  # Re-raise to be handled by caller
    
    def report_performance(self) -> None:
        """Report performance metrics using enhanced monitoring system."""
        try:
            # Check if it's time to report
            if self.performance_monitor.should_report():
                # Generate comprehensive performance report
                metrics = self.performance_monitor.report_performance()
                
                # Log additional system information
                summary = self.performance_monitor.get_performance_summary()
                if summary:
                    fps_stats = summary.get('fps', {})
                    memory_stats = summary.get('memory_mb', {})
                    
                    # Enhanced performance message
                    enhanced_msg = (
                        f"Enhanced Performance Report - "
                        f"FPS: {fps_stats.get('current', 0):.1f} "
                        f"(avg: {fps_stats.get('average', 0):.1f}, "
                        f"min: {fps_stats.get('min', 0):.1f}, "
                        f"max: {fps_stats.get('max', 0):.1f}) | "
                        f"Memory: {memory_stats.get('current', 0):.1f}MB "
                        f"(avg: {memory_stats.get('average', 0):.1f}MB) | "
                        f"Total frames: {summary.get('total_frames', 0)} | "
                        f"Uptime: {summary.get('uptime_seconds', 0):.1f}s"
                    )
                    
                    self.logger.info(enhanced_msg)
                    self.performance_logger.info(enhanced_msg)
                
                # Run health checks periodically
                health_results = self.health_checker.run_health_checks()
                failed_checks = [name for name, result in health_results.items() if not result]
                
                if failed_checks:
                    self.logger.warning(f"Health check failures: {failed_checks}")
                    # Log performance impact of failed components
                    if 'camera' in failed_checks:
                        self.logger.warning("Camera failure may impact frame processing performance")
                    if 'detector' in failed_checks:
                        self.logger.warning("Detector failure may impact detection performance")
                    if 'audio' in failed_checks:
                        self.logger.warning("Audio failure may impact user feedback responsiveness")
                
        except Exception as e:
            # Don't let performance reporting break the main loop
            self.logger.debug(f"Enhanced performance reporting error: {e}")
            # Fallback to basic reporting
            try:
                if self.start_time and self.frame_count > 0:
                    elapsed_time = time.time() - self.start_time
                    fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    self.logger.info(f"Basic Performance: {fps:.1f} FPS, {self.frame_count} frames")
            except Exception:
                pass  # Ignore fallback errors
    
    def run(self) -> None:
        """Main detection loop with comprehensive error handling and recovery."""
        with LoggingContextManager(self.logger, "main application run"):
            try:
                if not self.initialize():
                    raise SystemError("Failed to initialize system",
                                    ErrorSeverity.CRITICAL,
                                    "Check system logs for initialization errors")
                
                self.running = True
                self.logger.info("Starting main detection loop...")
                
                # Initialize performance tracking
                self.start_time = time.time()
                consecutive_failures = 0
                max_consecutive_failures = 10
                recovery_attempts = 0
                max_recovery_attempts = 3
                
                try:
                    while self.running:
                        try:
                            # Process single frame
                            frame_success = self.process_frame()
                            
                            if frame_success:
                                consecutive_failures = 0  # Reset failure counter on success
                            else:
                                consecutive_failures += 1
                                self.logger.warning(
                                    f"Frame processing failed "
                                    f"({consecutive_failures}/{max_consecutive_failures})"
                                )
                                
                                # If too many consecutive failures, attempt recovery
                                if consecutive_failures >= max_consecutive_failures:
                                    if recovery_attempts < max_recovery_attempts:
                                        self.logger.error(
                                            f"Too many consecutive failures, attempting system recovery "
                                            f"(attempt {recovery_attempts + 1}/{max_recovery_attempts})..."
                                        )
                                        
                                        if self.attempt_recovery():
                                            consecutive_failures = 0
                                            recovery_attempts += 1
                                            self.logger.info("System recovery successful")
                                        else:
                                            recovery_attempts += 1
                                            self.logger.error("System recovery failed")
                                            
                                            if recovery_attempts >= max_recovery_attempts:
                                                raise SystemError(
                                                    "Maximum recovery attempts exceeded",
                                                    ErrorSeverity.CRITICAL,
                                                    "System requires manual intervention"
                                                )
                                    else:
                                        raise SystemError(
                                            "System recovery attempts exhausted",
                                            ErrorSeverity.CRITICAL,
                                            "Restart application manually"
                                        )
                            
                            # Small delay to prevent excessive CPU usage
                            time.sleep(0.01)  # ~100 FPS max
                            
                        except KeyboardInterrupt:
                            self.logger.info("Keyboard interrupt received")
                            break
                            
                        except SystemError as e:
                            # System errors are already handled
                            self.error_handler.handle_error(e, "main loop")
                            break
                            
                        except Exception as e:
                            consecutive_failures += 1
                            
                            loop_error = SystemError(
                                f"Unexpected error in main loop: {e}",
                                ErrorSeverity.HIGH,
                                "Check system stability and restart if necessary"
                            )
                            self.error_handler.handle_error(loop_error, "main loop")
                            
                            if consecutive_failures >= max_consecutive_failures:
                                raise SystemError(
                                    "Too many consecutive main loop errors",
                                    ErrorSeverity.CRITICAL,
                                    "System is unstable, requires restart"
                                )
                            
                            # Brief pause before retry
                            time.sleep(0.1)
                    
                except SystemError as e:
                    # System errors that require shutdown
                    self.logger.critical(f"Critical system error: {e}")
                    
                except Exception as e:
                    # Unexpected fatal errors
                    fatal_error = SystemError(
                        f"Fatal error in main loop: {e}",
                        ErrorSeverity.CRITICAL,
                        "System encountered unexpected fatal error"
                    )
                    self.error_handler.handle_error(fatal_error, "main loop")
                    
            except SystemError as e:
                # Initialization or other system errors
                self.error_handler.handle_error(e, "application run")
                
            except Exception as e:
                # Catch-all for any other unexpected errors
                fatal_error = SystemError(
                    f"Unexpected application error: {e}",
                    ErrorSeverity.CRITICAL,
                    "Application encountered unexpected error during startup or execution"
                )
                self.error_handler.handle_error(fatal_error, "application run")
                
            finally:
                # Always cleanup, even if errors occurred
                self.cleanup()
    
    def attempt_recovery(self) -> bool:
        """
        Attempt to recover from system failures with comprehensive error handling.
        
        Returns:
            bool: True if recovery successful, False otherwise
        """
        with LoggingContextManager(self.logger, "system recovery"):
            try:
                recovery_success = True
                
                # Try to reinitialize camera
                if self.frame_processor:
                    try:
                        self.logger.info("Attempting camera recovery...")
                        safe_execute(self.frame_processor.release_camera, logger=self.logger)
                        
                        if not self.frame_processor.initialize_camera(
                            self.config.camera_index,
                            self.config.frame_width,
                            self.config.frame_height
                        ):
                            camera_error = CameraErrorCustom(
                                "Camera recovery failed during reinitialization",
                                ErrorSeverity.HIGH,
                                "Check camera hardware and connections"
                            )
                            self.error_handler.handle_error(camera_error, "camera recovery")
                            recovery_success = False
                        else:
                            self.logger.info("Camera recovery successful")
                            
                    except Exception as e:
                        camera_error = CameraErrorCustom(
                            f"Camera recovery error: {e}",
                            ErrorSeverity.HIGH,
                            "Camera hardware may need manual intervention"
                        )
                        self.error_handler.handle_error(camera_error, "camera recovery")
                        recovery_success = False
                
                # Try to reinitialize detector if needed
                if self.detector:
                    try:
                        self.logger.info("Checking detector status...")
                        model_info = safe_execute(
                            self.detector.get_model_info,
                            default_return={"status": "unknown"},
                            logger=self.logger
                        )
                        
                        if model_info.get("status") != "Model loaded":
                            self.logger.info("Attempting detector recovery...")
                            if not self.detector.load_model(self.config.model_name, self.config.device):
                                detection_error = DetectionError(
                                    "Detector recovery failed during model reload",
                                    ErrorSeverity.HIGH,
                                    "Check model file and GPU availability"
                                )
                                self.error_handler.handle_error(detection_error, "detector recovery")
                                recovery_success = False
                            else:
                                self.logger.info("Detector recovery successful")
                        else:
                            self.logger.info("Detector status is healthy, no recovery needed")
                            
                    except Exception as e:
                        detection_error = DetectionError(
                            f"Detector recovery error: {e}",
                            ErrorSeverity.HIGH,
                            "Model or GPU may need manual intervention"
                        )
                        self.error_handler.handle_error(detection_error, "detector recovery")
                        recovery_success = False
                
                # Try to reinitialize audio if needed
                if self.audio_manager:
                    try:
                        self.logger.info("Checking audio manager status...")
                        if not self.audio_manager.tts_engine:
                            self.logger.info("Attempting audio recovery...")
                            if not self.audio_manager.initialize_tts():
                                audio_error = AudioError(
                                    "Audio recovery failed during TTS reinitialization",
                                    ErrorSeverity.MEDIUM,
                                    "Audio functionality may be degraded"
                                )
                                self.error_handler.handle_error(audio_error, "audio recovery")
                                # Don't fail recovery for audio issues
                            else:
                                self.logger.info("Audio recovery successful")
                        else:
                            self.logger.info("Audio manager status is healthy")
                            
                    except Exception as e:
                        audio_error = AudioError(
                            f"Audio recovery error: {e}",
                            ErrorSeverity.MEDIUM,
                            "Audio system may need manual intervention"
                        )
                        self.error_handler.handle_error(audio_error, "audio recovery")
                        # Don't fail recovery for audio issues
                
                # Run health checks after recovery
                try:
                    health_results = self.health_checker.run_health_checks()
                    failed_checks = [name for name, result in health_results.items() if not result]
                    
                    if failed_checks:
                        self.logger.warning(f"Post-recovery health check failures: {failed_checks}")
                        
                        # Critical components must pass for successful recovery
                        critical_failures = [name for name in failed_checks 
                                           if name in ['camera', 'detector', 'system']]
                        if critical_failures:
                            self.logger.error(f"Critical components failed post-recovery: {critical_failures}")
                            recovery_success = False
                    else:
                        self.logger.info("All health checks passed after recovery")
                        
                except Exception as e:
                    self.logger.warning(f"Post-recovery health check failed: {e}")
                
                if recovery_success:
                    self.logger.info("System recovery completed successfully")
                    
                    # Announce recovery if audio is available
                    if self.audio_manager:
                        safe_execute(
                            lambda: self.audio_manager.announce("System recovered"),
                            logger=self.logger
                        )
                else:
                    self.logger.error("System recovery failed")
                
                return recovery_success
                
            except Exception as e:
                recovery_error = SystemError(
                    f"Recovery attempt failed with unexpected error: {e}",
                    ErrorSeverity.HIGH,
                    "Manual system restart may be required"
                )
                self.error_handler.handle_error(recovery_error, "system recovery")
                return False
    
    def cleanup(self) -> None:
        """Clean up resources and shutdown gracefully with comprehensive error handling."""
        with LoggingContextManager(self.logger, "system cleanup"):
            self.running = False
            
            try:
                # Announce shutdown if audio is available
                if self.audio_manager:
                    safe_execute(
                        lambda: self.audio_manager.announce("System shutting down"),
                        logger=self.logger
                    )
                    time.sleep(1.0)  # Give time for announcement
                
                # Release camera resources
                if self.frame_processor:
                    self.logger.info("Releasing camera resources...")
                    safe_execute(
                        self.frame_processor.release_camera,
                        logger=self.logger
                    )
                
                # Cleanup audio manager
                if self.audio_manager:
                    self.logger.info("Cleaning up audio manager...")
                    safe_execute(
                        self.audio_manager.cleanup,
                        logger=self.logger
                    )
                
                # Final performance report
                if self.start_time and self.frame_count > 0:
                    total_time = time.time() - self.start_time
                    avg_fps = self.frame_count / total_time if total_time > 0 else 0
                    
                    performance_summary = (
                        f"Final performance: {avg_fps:.1f} FPS average, "
                        f"{self.frame_count} total frames in {total_time:.1f} seconds"
                    )
                    
                    self.logger.info(performance_summary)
                    self.performance_logger.info(performance_summary)
                
                # Log error summary
                error_summary = self.error_handler.get_error_summary()
                if error_summary['total_errors'] > 0:
                    self.logger.info(f"Session error summary: {error_summary['total_errors']} total errors")
                    for error_type, count in error_summary['error_counts'].items():
                        self.logger.info(f"  {error_type}: {count} occurrences")
                
                # Shutdown logging system
                shutdown_logging()
                
                self.logger.info("System shutdown completed successfully")
                
            except Exception as e:
                # Use print as fallback since logging might be shutting down
                print(f"Error during cleanup: {e}")
                try:
                    self.logger.error(f"Error during cleanup: {e}")
                except:
                    pass
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def get_system_status(self) -> dict:
        """
        Get current system status information.
        
        Returns:
            dict: System status information
        """
        status = {
            "initialized": self.initialized,
            "running": self.running,
            "frame_count": self.frame_count,
            "components": {}
        }
        
        # Get component status
        if self.frame_processor:
            status["components"]["camera"] = self.frame_processor.get_camera_info()
        
        if self.detector:
            status["components"]["detector"] = self.detector.get_model_info()
        
        if self.audio_manager:
            status["components"]["audio"] = {
                "initialized": self.audio_manager.tts_engine is not None,
                "speaking": self.audio_manager.is_speaking
            }
        
        # Performance metrics
        if self.start_time:
            elapsed = time.time() - self.start_time
            status["performance"] = {
                "elapsed_time": elapsed,
                "average_fps": self.frame_count / elapsed if elapsed > 0 else 0
            }
        
        return status
    
    def run_single_detection(self) -> dict:
        """
        Run a single detection cycle for testing purposes.
        
        Returns:
            dict: Detection results and status
        """
        if not self.initialized:
            return {"error": "System not initialized"}
        
        try:
            # Capture and process single frame
            raw_frame = self.frame_processor.capture_frame()
            if raw_frame is None:
                return {"error": "Failed to capture frame"}
            
            processed_frame = self.frame_processor.preprocess_frame(raw_frame)
            detections = self.detector.detect_objects(
                processed_frame, 
                confidence_threshold=self.config.confidence_threshold
            )
            
            # Get spatial analysis
            results = []
            if detections:
                prioritized = self.spatial_analyzer.prioritize_objects(detections)
                for detection in prioritized:
                    position = self.spatial_analyzer.calculate_position(detection)
                    distance = self.spatial_analyzer.estimate_distance(detection)
                    results.append({
                        "object": detection.class_name,
                        "confidence": detection.confidence,
                        "position": position,
                        "distance": distance,
                        "bbox": detection.bbox
                    })
            
            return {
                "success": True,
                "detections_count": len(detections),
                "results": results,
                "frame_shape": processed_frame.shape
            }
            
        except Exception as e:
            return {"error": f"Single detection failed: {e}"}


def main():
    """Entry point for the application with comprehensive error handling."""
    app = None
    
    try:
        print("Starting Enhanced Blind Detection System...")
        
        # Initialize application
        app = BlindDetectionApp()
        
        # Run main application
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        if app and app.logger:
            app.logger.info("Application interrupted by user")
            
    except SystemError as e:
        print(f"System error: {e}")
        if e.recovery_suggestion:
            print(f"Suggestion: {e.recovery_suggestion}")
        if app and app.logger:
            app.logger.critical(f"System error during startup: {e}")
            
    except (CameraErrorCustom, DetectionError, AudioError) as e:
        print(f"Component error: {e}")
        if e.recovery_suggestion:
            print(f"Suggestion: {e.recovery_suggestion}")
        if app and app.logger:
            app.logger.error(f"Component error during startup: {e}")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check the logs for more details")
        
        # Try to log the error if possible
        try:
            if app and app.logger:
                app.logger.critical(f"Unexpected application error: {e}", exc_info=True)
            else:
                # Fallback logging
                import logging
                logging.basicConfig(level=logging.ERROR)
                logging.error(f"Application startup error: {e}", exc_info=True)
        except:
            # If logging fails, just print
            import traceback
            print("Logging failed. Full traceback:")
            traceback.print_exc()
            
    finally:
        print("Application terminated")
        
        # Ensure cleanup happens even if app creation failed
        if app:
            try:
                if hasattr(app, 'cleanup'):
                    app.cleanup()
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")
        
        # Final logging shutdown
        try:
            shutdown_logging()
        except:
            pass


if __name__ == "__main__":
    main()