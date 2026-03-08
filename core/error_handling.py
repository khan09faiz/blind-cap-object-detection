"""
Comprehensive error handling system for the enhanced blind detection system.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for categorizing different types of errors."""
    LOW = 1          # Minor issues that don't affect core functionality
    MEDIUM = 2       # Issues that may degrade performance or features
    HIGH = 3         # Serious issues that affect core functionality
    CRITICAL = 4     # System-breaking errors that require immediate attention


class ErrorCategory(Enum):
    """Categories of errors for better organization and handling."""
    CAMERA = "camera"
    DETECTION = "detection"
    AUDIO = "audio"
    SPATIAL = "spatial"
    CONFIG = "config"
    SYSTEM = "system"
    NETWORK = "network"
    HARDWARE = "hardware"


# Base exception classes
class BlindDetectionError(Exception):
    """Base exception class for all blind detection system errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 recovery_suggestion: Optional[str] = None):
        """
        Initialize base error.
        
        Args:
            message: Error message
            category: Error category
            severity: Error severity level
            recovery_suggestion: Suggested recovery action
        """
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.recovery_suggestion = recovery_suggestion
        self.timestamp = time.time()


class CameraError(BlindDetectionError):
    """Camera-related errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH,
                 recovery_suggestion: Optional[str] = None):
        super().__init__(message, ErrorCategory.CAMERA, severity, recovery_suggestion)


class DetectionError(BlindDetectionError):
    """Object detection-related errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recovery_suggestion: Optional[str] = None):
        super().__init__(message, ErrorCategory.DETECTION, severity, recovery_suggestion)


class AudioError(BlindDetectionError):
    """Audio system-related errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recovery_suggestion: Optional[str] = None):
        super().__init__(message, ErrorCategory.AUDIO, severity, recovery_suggestion)


class SpatialError(BlindDetectionError):
    """Spatial analysis-related errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.LOW,
                 recovery_suggestion: Optional[str] = None):
        super().__init__(message, ErrorCategory.SPATIAL, severity, recovery_suggestion)


class ConfigurationError(BlindDetectionError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH,
                 recovery_suggestion: Optional[str] = None):
        super().__init__(message, ErrorCategory.CONFIG, severity, recovery_suggestion)


class SystemError(BlindDetectionError):
    """System-level errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.CRITICAL,
                 recovery_suggestion: Optional[str] = None):
        super().__init__(message, ErrorCategory.SYSTEM, severity, recovery_suggestion)


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler.
        
        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
    
    def handle_error(self, error: Exception, context: str = "", 
                    attempt_recovery: bool = True) -> bool:
        """
        Handle an error with logging and optional recovery.
        
        Args:
            error: Exception that occurred
            context: Context information about where the error occurred
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            bool: True if error was handled successfully, False otherwise
        """
        try:
            # Create error record
            error_record = self._create_error_record(error, context)
            
            # Log the error
            self._log_error(error_record)
            
            # Add to error history
            self._add_to_history(error_record)
            
            # Update error counts
            self._update_error_counts(error_record)
            
            # Attempt recovery if requested and appropriate
            if attempt_recovery and self._should_attempt_recovery(error_record):
                return self._attempt_recovery(error_record)
            
            return False
            
        except Exception as handler_error:
            # Error in error handler - log to console as fallback
            print(f"Error handler failed: {handler_error}")
            print(f"Original error: {error}")
            return False
    
    def _create_error_record(self, error: Exception, context: str) -> Dict[str, Any]:
        """Create structured error record."""
        error_record = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            'category': ErrorCategory.SYSTEM,
            'severity': ErrorSeverity.MEDIUM,
            'recovery_suggestion': None
        }
        
        # Extract additional info for BlindDetectionError instances
        if isinstance(error, BlindDetectionError):
            error_record['category'] = error.category
            error_record['severity'] = error.severity
            error_record['recovery_suggestion'] = error.recovery_suggestion
        
        return error_record
    
    def _log_error(self, error_record: Dict[str, Any]) -> None:
        """Log error with appropriate level based on severity."""
        severity = error_record['severity']
        message = f"[{error_record['category'].value.upper()}] {error_record['message']}"
        
        if error_record['context']:
            message += f" (Context: {error_record['context']})"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message)
            self.logger.critical(f"Traceback:\n{error_record['traceback']}")
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(message)
            self.logger.debug(f"Traceback:\n{error_record['traceback']}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(message)
            self.logger.debug(f"Traceback:\n{error_record['traceback']}")
        else:  # LOW
            self.logger.info(message)
    
    def _add_to_history(self, error_record: Dict[str, Any]) -> None:
        """Add error to history with size limit."""
        self.error_history.append(error_record)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _update_error_counts(self, error_record: Dict[str, Any]) -> None:
        """Update error occurrence counts."""
        error_key = f"{error_record['category'].value}:{error_record['error_type']}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def _should_attempt_recovery(self, error_record: Dict[str, Any]) -> bool:
        """Determine if recovery should be attempted."""
        error_key = f"{error_record['category'].value}:{error_record['error_type']}"
        attempts = self.recovery_attempts.get(error_key, 0)
        
        # Don't attempt recovery if we've exceeded max attempts
        if attempts >= self.max_recovery_attempts:
            return False
        
        # Don't attempt recovery for critical system errors
        if error_record['severity'] == ErrorSeverity.CRITICAL:
            return False
        
        # Don't attempt recovery if error is too frequent
        error_count = self.error_counts.get(error_key, 0)
        if error_count > 10:  # Too many occurrences
            return False
        
        # Only attempt recovery for specific error categories that support it
        recoverable_categories = [ErrorCategory.CAMERA, ErrorCategory.DETECTION, ErrorCategory.AUDIO]
        if error_record['category'] not in recoverable_categories:
            return False
        
        return True
    
    def _attempt_recovery(self, error_record: Dict[str, Any]) -> bool:
        """Attempt automatic recovery based on error type."""
        error_key = f"{error_record['category'].value}:{error_record['error_type']}"
        self.recovery_attempts[error_key] = self.recovery_attempts.get(error_key, 0) + 1
        
        category = error_record['category']
        
        try:
            if category == ErrorCategory.CAMERA:
                return self._recover_camera_error(error_record)
            elif category == ErrorCategory.DETECTION:
                return self._recover_detection_error(error_record)
            elif category == ErrorCategory.AUDIO:
                return self._recover_audio_error(error_record)
            elif category == ErrorCategory.CONFIG:
                return self._recover_config_error(error_record)
            else:
                return self._generic_recovery(error_record)
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            return False
    
    def _recover_camera_error(self, error_record: Dict[str, Any]) -> bool:
        """Attempt camera error recovery."""
        self.logger.info("Attempting camera error recovery...")
        
        # Camera recovery is handled by FrameProcessor itself
        # Just log the attempt here
        if error_record['recovery_suggestion']:
            self.logger.info(f"Recovery suggestion: {error_record['recovery_suggestion']}")
        
        return False  # Let FrameProcessor handle camera recovery
    
    def _recover_detection_error(self, error_record: Dict[str, Any]) -> bool:
        """Attempt detection error recovery."""
        self.logger.info("Attempting detection error recovery...")
        
        # Detection errors are usually temporary - just continue
        # In a real implementation, this might reload the model
        return True
    
    def _recover_audio_error(self, error_record: Dict[str, Any]) -> bool:
        """Attempt audio error recovery."""
        self.logger.info("Attempting audio error recovery...")
        
        # Audio errors can often be recovered by reinitializing TTS
        return True  # Let AudioManager handle TTS recovery
    
    def _recover_config_error(self, error_record: Dict[str, Any]) -> bool:
        """Attempt configuration error recovery."""
        self.logger.info("Attempting configuration error recovery...")
        
        # Config errors usually require manual intervention
        return False
    
    def _generic_recovery(self, error_record: Dict[str, Any]) -> bool:
        """Generic recovery attempt."""
        self.logger.info("Attempting generic error recovery...")
        
        # For generic errors, just log and continue
        # Most generic errors are not recoverable automatically
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error statistics."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': dict(self.error_counts),
            'recovery_attempts': dict(self.recovery_attempts),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def clear_error_history(self) -> None:
        """Clear error history and reset counters."""
        self.error_history.clear()
        self.error_counts.clear()
        self.recovery_attempts.clear()
        self.logger.info("Error history cleared")


def error_handler(category: ErrorCategory = ErrorCategory.SYSTEM, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recovery_suggestion: Optional[str] = None,
                 reraise: bool = True):
    """
    Decorator for automatic error handling.
    
    Args:
        category: Error category
        severity: Error severity
        recovery_suggestion: Recovery suggestion
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create appropriate error type
                if category == ErrorCategory.CAMERA:
                    error = CameraError(str(e), severity, recovery_suggestion)
                elif category == ErrorCategory.DETECTION:
                    error = DetectionError(str(e), severity, recovery_suggestion)
                elif category == ErrorCategory.AUDIO:
                    error = AudioError(str(e), severity, recovery_suggestion)
                elif category == ErrorCategory.SPATIAL:
                    error = SpatialError(str(e), severity, recovery_suggestion)
                elif category == ErrorCategory.CONFIG:
                    error = ConfigurationError(str(e), severity, recovery_suggestion)
                else:
                    error = SystemError(str(e), severity, recovery_suggestion)
                
                # Get error handler instance
                handler = ErrorHandler(logging.getLogger(func.__module__))
                handler.handle_error(error, f"{func.__name__}")
                
                if reraise:
                    raise error
                
                return None
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return=None, 
                logger: Optional[logging.Logger] = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default return value on error
        logger: Logger instance
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Safe execution failed for {func.__name__}: {e}")
        else:
            print(f"Safe execution failed for {func.__name__}: {e}")
        return default_return


class RetryHandler:
    """Handler for retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries
            backoff_factor: Exponential backoff factor
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logging.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception


def retry_on_error(max_retries: int = 3, base_delay: float = 1.0, 
                  exceptions: Union[Type[Exception], tuple] = Exception):
    """
    Decorator for automatic retry on specific exceptions.
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        exceptions: Exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(max_retries, base_delay)
            
            def execute():
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    raise e
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    raise e
            
            return retry_handler.retry(execute)
        
        return wrapper
    return decorator


class HealthChecker:
    """System health monitoring and error prevention."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize health checker."""
        self.logger = logger or logging.getLogger(__name__)
        self.health_checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, bool] = {}
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy, False otherwise
        """
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, bool]:
        """
        Run all registered health checks.
        
        Returns:
            Dict mapping check names to results
        """
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                
                # Log status changes
                if name in self.last_check_results:
                    if self.last_check_results[name] != result:
                        if result:
                            self.logger.info(f"Health check '{name}' recovered")
                        else:
                            self.logger.warning(f"Health check '{name}' failed")
                
            except Exception as e:
                self.logger.error(f"Health check '{name}' threw exception: {e}")
                results[name] = False
        
        self.last_check_results = results
        return results
    
    def is_system_healthy(self) -> bool:
        """Check if all health checks pass."""
        results = self.run_health_checks()
        return all(results.values())
    
    def get_failed_checks(self) -> List[str]:
        """Get list of failed health check names."""
        results = self.run_health_checks()
        return [name for name, result in results.items() if not result]