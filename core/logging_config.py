"""
Centralized logging configuration for the enhanced blind detection system.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional


class LoggingManager:
    """Centralized logging management with file rotation and error handling."""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize logging manager.
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.loggers = {}
        
        # Create logs directory if it doesn't exist
        self._ensure_log_directory()
        
        # Setup root logger configuration
        self._setup_root_logger()
    
    def _ensure_log_directory(self) -> None:
        """Create log directory if it doesn't exist."""
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create log directory {self.log_dir}: {e}")
            # Fallback to current directory
            self.log_dir = "."
            try:
                # Ensure current directory is writable
                test_file = os.path.join(self.log_dir, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as fallback_error:
                print(f"Warning: Current directory not writable: {fallback_error}")
                # Last resort - use temp directory
                import tempfile
                self.log_dir = tempfile.gettempdir()
    
    def _setup_root_logger(self) -> None:
        """Setup root logger with file and console handlers."""
        try:
            # Clear any existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            
            # Set root logger level
            root_logger.setLevel(self.log_level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Setup file handler with rotation
            log_file = os.path.join(self.log_dir, 'blind_detection.log')
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            
            # Setup console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)  # Console shows INFO and above
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            
            # Add handlers to root logger
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
            # Log startup message
            logging.info("Logging system initialized successfully")
            logging.info(f"Log level: {logging.getLevelName(self.log_level)}")
            logging.info(f"Log directory: {os.path.abspath(self.log_dir)}")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            # Setup minimal console logging as fallback
            logging.basicConfig(
                level=self.log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                stream=sys.stdout
            )
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger for a specific module.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def setup_error_file_handler(self, logger_name: str) -> None:
        """
        Setup separate error file handler for critical errors.
        
        Args:
            logger_name: Name of the logger to add error handler to
        """
        try:
            logger = self.get_logger(logger_name)
            
            # Create error-specific file handler
            error_file = os.path.join(self.log_dir, 'errors.log')
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            
            # Error-specific formatter with more detail
            error_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d\n'
                'Message: %(message)s\n'
                'Process: %(process)d - Thread: %(thread)d\n'
                '---\n',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            error_handler.setFormatter(error_formatter)
            
            logger.addHandler(error_handler)
            
        except Exception as e:
            logging.warning(f"Could not setup error file handler: {e}")
    
    def log_system_info(self) -> None:
        """Log system information for debugging purposes."""
        try:
            import platform
            import psutil
            
            logger = self.get_logger('system_info')
            
            logger.info("=== System Information ===")
            logger.info(f"Platform: {platform.platform()}")
            logger.info(f"Python version: {platform.python_version()}")
            logger.info(f"CPU count: {psutil.cpu_count()}")
            logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
            
            # GPU information if available
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
                    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
                else:
                    logger.info("CUDA not available")
            except ImportError:
                logger.info("PyTorch not available for GPU detection")
            
            logger.info("=== End System Information ===")
            
        except Exception as e:
            logging.warning(f"Could not log system information: {e}")
    
    def create_performance_logger(self) -> logging.Logger:
        """
        Create specialized logger for performance metrics.
        
        Returns:
            logging.Logger: Performance logger
        """
        try:
            perf_logger = self.get_logger('performance')
            
            # Create performance-specific file handler
            perf_file = os.path.join(self.log_dir, 'performance.log')
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_file,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=2,
                encoding='utf-8'
            )
            perf_handler.setLevel(logging.INFO)
            
            # Performance-specific formatter
            perf_formatter = logging.Formatter(
                '%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            perf_handler.setFormatter(perf_formatter)
            
            perf_logger.addHandler(perf_handler)
            
            return perf_logger
            
        except Exception as e:
            logging.warning(f"Could not create performance logger: {e}")
            return self.get_logger('performance')
    
    def log_exception(self, logger: logging.Logger, message: str, exc_info: bool = True) -> None:
        """
        Log exception with full traceback.
        
        Args:
            logger: Logger instance to use
            message: Error message
            exc_info: Whether to include exception info
        """
        try:
            logger.error(message, exc_info=exc_info)
        except Exception as e:
            # Fallback logging if main logger fails
            print(f"Logging failed: {e}")
            print(f"Original message: {message}")
    
    def shutdown(self) -> None:
        """Shutdown logging system gracefully."""
        try:
            logging.info("Shutting down logging system...")
            
            # Close all handlers
            for logger in self.loggers.values():
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
            
            # Close root logger handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
                
        except Exception as e:
            print(f"Error during logging shutdown: {e}")


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def initialize_logging(log_dir: str = "logs", log_level: str = "INFO") -> LoggingManager:
    """
    Initialize global logging manager.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        LoggingManager: Initialized logging manager
    """
    global _logging_manager
    
    if _logging_manager is None:
        _logging_manager = LoggingManager(log_dir, log_level)
        _logging_manager.log_system_info()
    
    return _logging_manager


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance from global logging manager.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    global _logging_manager
    
    if _logging_manager is None:
        _logging_manager = initialize_logging()
    
    return _logging_manager.get_logger(name)


def shutdown_logging() -> None:
    """Shutdown global logging manager."""
    global _logging_manager
    
    if _logging_manager is not None:
        _logging_manager.shutdown()
        _logging_manager = None


class LoggingContextManager:
    """Context manager for logging operations with automatic error handling."""
    
    def __init__(self, logger: logging.Logger, operation: str, log_level: int = logging.INFO):
        """
        Initialize logging context manager.
        
        Args:
            logger: Logger instance
            operation: Description of the operation
            log_level: Log level for start/end messages
        """
        self.logger = logger
        self.operation = operation
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        """Enter context - log operation start."""
        import time
        self.start_time = time.time()
        self.logger.log(self.log_level, f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log operation end and handle exceptions."""
        import time
        
        if self.start_time:
            duration = time.time() - self.start_time
            
            if exc_type is None:
                self.logger.log(self.log_level, f"Completed {self.operation} in {duration:.3f}s")
            else:
                self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
        
        # Don't suppress exceptions
        return False


def log_performance(func):
    """Decorator for logging function performance."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper