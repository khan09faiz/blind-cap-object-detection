"""
Performance monitoring and optimization module for Enhanced Blind Detection System.
Provides FPS monitoring, performance metrics, benchmarking, and optimization features.
"""

import time
import psutil
import threading
import statistics
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import os
import gc
import tracemalloc
import logging
from contextlib import contextmanager

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    timestamp: float
    fps: float
    frame_time_ms: float
    detection_time_ms: float
    audio_time_ms: float
    total_time_ms: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    frame_count: int = 0
    detection_count: int = 0


@dataclass
class SystemInfo:
    """System information for performance analysis."""
    cpu_count: int
    cpu_freq_mhz: float
    total_memory_gb: float
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    cuda_available: bool = False
    opencv_version: Optional[str] = None
    torch_version: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results from performance benchmarking."""
    test_name: str
    duration_seconds: float
    frames_processed: int
    avg_fps: float
    min_fps: float
    max_fps: float
    avg_frame_time_ms: float
    avg_detection_time_ms: float
    avg_memory_mb: float
    peak_memory_mb: float
    cpu_utilization_percent: float
    success_rate: float
    errors: List[str] = field(default_factory=list)


class PerformanceMonitor:
    """Advanced performance monitoring system."""
    
    def __init__(self, window_size: int = 100, report_interval: float = 30.0):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of measurements to keep for rolling averages
            report_interval: Seconds between performance reports
        """
        self.window_size = window_size
        self.report_interval = report_interval
        
        # Performance data storage
        self.metrics_history: deque = deque(maxlen=window_size)
        self.frame_times: deque = deque(maxlen=window_size)
        self.detection_times: deque = deque(maxlen=window_size)
        self.audio_times: deque = deque(maxlen=window_size)
        
        # Timing data
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.frame_count = 0
        self.detection_count = 0
        
        # Current measurement
        self.frame_start_time = None
        self.detection_start_time = None
        self.audio_start_time = None
        
        # System monitoring
        self.process = psutil.Process()
        self.system_info = self._get_system_info()
        
        # Threading
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Performance logger
        self.logger = logging.getLogger('performance')
        
        # Memory tracking
        self.memory_tracking = False
        
    def _get_system_info(self) -> SystemInfo:
        """Get system information for performance analysis."""
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        
        info = SystemInfo(
            cpu_count=psutil.cpu_count(),
            cpu_freq_mhz=cpu_freq.current if cpu_freq else 0,
            total_memory_gb=memory.total / (1024**3)
        )
        
        # GPU information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info.cuda_available = True
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info.torch_version = torch.__version__
        
        # OpenCV version
        if CV2_AVAILABLE:
            info.opencv_version = cv2.__version__
        
        return info
    
    def start_memory_tracking(self) -> None:
        """Start memory tracking using tracemalloc."""
        if not self.memory_tracking:
            tracemalloc.start()
            self.memory_tracking = True
            self.logger.info("Memory tracking started")
    
    def stop_memory_tracking(self) -> None:
        """Stop memory tracking."""
        if self.memory_tracking:
            tracemalloc.stop()
            self.memory_tracking = False
            self.logger.info("Memory tracking stopped")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if self.memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            return {
                'current_mb': current / (1024 * 1024),
                'peak_mb': peak / (1024 * 1024)
            }
        return {'current_mb': 0, 'peak_mb': 0}
    
    @contextmanager
    def measure_frame(self):
        """Context manager for measuring frame processing time."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            frame_time = (end_time - start_time) * 1000  # Convert to ms
            
            with self.lock:
                self.frame_times.append(frame_time)
                self.frame_count += 1
    
    @contextmanager  
    def measure_detection(self):
        """Context manager for measuring detection time."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            detection_time = (end_time - start_time) * 1000  # Convert to ms
            
            with self.lock:
                self.detection_times.append(detection_time)
                self.detection_count += 1
    
    @contextmanager
    def measure_audio(self):
        """Context manager for measuring audio processing time."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            audio_time = (end_time - start_time) * 1000  # Convert to ms
            
            with self.lock:
                self.audio_times.append(audio_time)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate FPS
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Get timing averages
        avg_frame_time = statistics.mean(self.frame_times) if self.frame_times else 0
        avg_detection_time = statistics.mean(self.detection_times) if self.detection_times else 0
        avg_audio_time = statistics.mean(self.audio_times) if self.audio_times else 0
        
        # System metrics
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # GPU memory if available
        gpu_memory_mb = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass
        
        return PerformanceMetrics(
            timestamp=current_time,
            fps=fps,
            frame_time_ms=avg_frame_time,
            detection_time_ms=avg_detection_time,
            audio_time_ms=avg_audio_time,
            total_time_ms=avg_frame_time + avg_detection_time + avg_audio_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            frame_count=self.frame_count,
            detection_count=self.detection_count
        )
    
    def should_report(self) -> bool:
        """Check if it's time to report performance."""
        current_time = time.time()
        return (current_time - self.last_report_time) >= self.report_interval
    
    def report_performance(self) -> PerformanceMetrics:
        """Generate and log performance report."""
        metrics = self.get_current_metrics()
        
        # Store metrics
        with self.lock:
            self.metrics_history.append(metrics)
        
        # Log performance report
        self.logger.info(
            f"Performance Report - FPS: {metrics.fps:.1f}, "
            f"Frame Time: {metrics.frame_time_ms:.1f}ms, "
            f"Detection Time: {metrics.detection_time_ms:.1f}ms, "
            f"CPU: {metrics.cpu_percent:.1f}%, "
            f"Memory: {metrics.memory_mb:.1f}MB"
        )
        
        if metrics.gpu_memory_mb:
            self.logger.info(f"GPU Memory: {metrics.gpu_memory_mb:.1f}MB")
        
        self.last_report_time = metrics.timestamp
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {}
        
        fps_values = [m.fps for m in self.metrics_history]
        frame_times = [m.frame_time_ms for m in self.metrics_history]
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_mb for m in self.metrics_history]
        
        return {
            'fps': {
                'current': fps_values[-1],
                'average': statistics.mean(fps_values),
                'min': min(fps_values),
                'max': max(fps_values),
                'std_dev': statistics.stdev(fps_values) if len(fps_values) > 1 else 0
            },
            'frame_time_ms': {
                'current': frame_times[-1],
                'average': statistics.mean(frame_times),
                'min': min(frame_times),
                'max': max(frame_times)
            },
            'cpu_percent': {
                'current': cpu_values[-1],
                'average': statistics.mean(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory_mb': {
                'current': memory_values[-1],
                'average': statistics.mean(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'total_frames': self.frame_count,
            'total_detections': self.detection_count,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def reset_counters(self) -> None:
        """Reset performance counters."""
        with self.lock:
            self.start_time = time.time()
            self.last_report_time = self.start_time
            self.frame_count = 0
            self.detection_count = 0
            self.frame_times.clear()
            self.detection_times.clear()
            self.audio_times.clear()
            self.metrics_history.clear()
        
        self.logger.info("Performance counters reset")


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    def __init__(self, config=None):
        """
        Initialize performance optimizer.
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.logger = logging.getLogger('performance')
        self.optimizations_applied = []
    
    def optimize_torch(self) -> List[str]:
        """Apply PyTorch optimizations."""
        optimizations = []
        
        if not TORCH_AVAILABLE:
            return optimizations
        
        try:
            # Enable memory efficient attention if available
            if hasattr(torch.backends, 'cuda') and torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                optimizations.append("Enabled TensorFloat-32 for CUDA")
            
            # Set number of threads for CPU inference
            if torch.get_num_threads() != psutil.cpu_count():
                torch.set_num_threads(psutil.cpu_count())
                optimizations.append(f"Set PyTorch threads to {psutil.cpu_count()}")
            
            # Enable cuDNN if available
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                optimizations.append("Enabled cuDNN benchmarking")
                
        except Exception as e:
            self.logger.warning(f"PyTorch optimization error: {e}")
        
        return optimizations
    
    def optimize_opencv(self) -> List[str]:
        """Apply OpenCV optimizations."""
        optimizations = []
        
        if not CV2_AVAILABLE:
            return optimizations
        
        try:
            # Enable multi-threading
            num_threads = psutil.cpu_count()
            cv2.setNumThreads(num_threads)
            optimizations.append(f"Set OpenCV threads to {num_threads}")
            
            # Use optimized code paths if available
            if cv2.useOptimized():
                optimizations.append("OpenCV optimized code paths enabled")
            else:
                cv2.setUseOptimized(True)
                optimizations.append("Enabled OpenCV optimizations")
                
        except Exception as e:
            self.logger.warning(f"OpenCV optimization error: {e}")
        
        return optimizations
    
    def optimize_memory(self) -> List[str]:
        """Apply memory optimizations."""
        optimizations = []
        
        try:
            # Force garbage collection
            collected = gc.collect()
            if collected > 0:
                optimizations.append(f"Collected {collected} garbage objects")
            
            # Set garbage collection thresholds
            gc.set_threshold(700, 10, 10)
            optimizations.append("Optimized garbage collection thresholds")
            
            # Clear GPU cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimizations.append("Cleared GPU memory cache")
                
        except Exception as e:
            self.logger.warning(f"Memory optimization error: {e}")
        
        return optimizations
    
    def optimize_system(self) -> List[str]:
        """Apply system-level optimizations."""
        optimizations = []
        
        try:
            # Set process priority (if possible)
            try:
                process = psutil.Process()
                if process.nice() != psutil.ABOVE_NORMAL_PRIORITY_CLASS:
                    process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                    optimizations.append("Increased process priority")
            except (AttributeError, PermissionError):
                pass
            
            # CPU affinity optimization for multi-core systems
            cpu_count = psutil.cpu_count()
            if cpu_count > 4:
                # Use performance cores (first 75% of CPUs)
                performance_cores = list(range(int(cpu_count * 0.75)))
                try:
                    psutil.Process().cpu_affinity(performance_cores)
                    optimizations.append(f"Set CPU affinity to performance cores: {performance_cores}")
                except (AttributeError, PermissionError):
                    pass
                    
        except Exception as e:
            self.logger.warning(f"System optimization error: {e}")
        
        return optimizations
    
    def apply_all_optimizations(self) -> Dict[str, List[str]]:
        """Apply all available optimizations."""
        optimizations = {
            'torch': self.optimize_torch(),
            'opencv': self.optimize_opencv(),
            'memory': self.optimize_memory(),
            'system': self.optimize_system()
        }
        
        # Log optimizations
        total_optimizations = sum(len(opts) for opts in optimizations.values())
        self.logger.info(f"Applied {total_optimizations} performance optimizations")
        
        for category, opts in optimizations.items():
            for opt in opts:
                self.logger.debug(f"{category.upper()}: {opt}")
        
        self.optimizations_applied = optimizations
        return optimizations


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize benchmark system.
        
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor
        self.logger = logging.getLogger('performance')
    
    def run_fps_benchmark(self, duration_seconds: float = 60.0, 
                         target_fps: Optional[float] = None) -> BenchmarkResult:
        """
        Run FPS benchmark test.
        
        Args:
            duration_seconds: Duration of benchmark
            target_fps: Target FPS for comparison
            
        Returns:
            BenchmarkResult with benchmark data
        """
        self.logger.info(f"Starting FPS benchmark for {duration_seconds} seconds")
        
        start_time = time.time()
        fps_samples = []
        frame_time_samples = []
        memory_samples = []
        errors = []
        
        # Reset counters
        self.monitor.reset_counters()
        
        while (time.time() - start_time) < duration_seconds:
            try:
                # Simulate frame processing
                with self.monitor.measure_frame():
                    time.sleep(0.001)  # Simulate minimal processing
                
                # Collect metrics
                metrics = self.monitor.get_current_metrics()
                fps_samples.append(metrics.fps)
                frame_time_samples.append(metrics.frame_time_ms)
                memory_samples.append(metrics.memory_mb)
                
            except Exception as e:
                errors.append(str(e))
        
        # Calculate results
        end_time = time.time()
        actual_duration = end_time - start_time
        
        if fps_samples:
            avg_fps = statistics.mean(fps_samples)
            min_fps = min(fps_samples)
            max_fps = max(fps_samples)
        else:
            avg_fps = min_fps = max_fps = 0
        
        if frame_time_samples:
            avg_frame_time = statistics.mean(frame_time_samples)
        else:
            avg_frame_time = 0
        
        if memory_samples:
            avg_memory = statistics.mean(memory_samples)
            peak_memory = max(memory_samples)
        else:
            avg_memory = peak_memory = 0
        
        # Calculate success rate
        success_rate = (len(fps_samples) / max(1, self.monitor.frame_count)) * 100
        
        result = BenchmarkResult(
            test_name="FPS Benchmark",
            duration_seconds=actual_duration,
            frames_processed=self.monitor.frame_count,
            avg_fps=avg_fps,
            min_fps=min_fps,
            max_fps=max_fps,
            avg_frame_time_ms=avg_frame_time,
            avg_detection_time_ms=0,  # Not measured in this test
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            cpu_utilization_percent=self.monitor.process.cpu_percent(),
            success_rate=success_rate,
            errors=errors
        )
        
        self.logger.info(f"FPS benchmark completed: {avg_fps:.1f} FPS average")
        return result
    
    def run_memory_benchmark(self, iterations: int = 1000) -> BenchmarkResult:
        """
        Run memory usage benchmark.
        
        Args:
            iterations: Number of iterations to run
            
        Returns:
            BenchmarkResult with memory benchmark data
        """
        self.logger.info(f"Starting memory benchmark with {iterations} iterations")
        
        start_time = time.time()
        memory_samples = []
        errors = []
        
        # Start memory tracking
        self.monitor.start_memory_tracking()
        
        try:
            for i in range(iterations):
                try:
                    # Simulate memory-intensive operations
                    data = [list(range(1000)) for _ in range(100)]
                    
                    # Force garbage collection occasionally
                    if i % 100 == 0:
                        gc.collect()
                    
                    # Collect memory stats
                    memory_stats = self.monitor.get_memory_stats()
                    memory_samples.append(memory_stats['current_mb'])
                    
                    del data  # Clean up
                    
                except Exception as e:
                    errors.append(str(e))
        
        finally:
            self.monitor.stop_memory_tracking()
        
        end_time = time.time()
        
        # Calculate results
        if memory_samples:
            avg_memory = statistics.mean(memory_samples)
            peak_memory = max(memory_samples)
        else:
            avg_memory = peak_memory = 0
        
        success_rate = (len(memory_samples) / iterations) * 100
        
        result = BenchmarkResult(
            test_name="Memory Benchmark",
            duration_seconds=end_time - start_time,
            frames_processed=iterations,
            avg_fps=0,  # Not applicable
            min_fps=0,
            max_fps=0,
            avg_frame_time_ms=0,
            avg_detection_time_ms=0,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            cpu_utilization_percent=self.monitor.process.cpu_percent(),
            success_rate=success_rate,
            errors=errors
        )
        
        self.logger.info(f"Memory benchmark completed: {avg_memory:.1f}MB average")
        return result
    
    def save_benchmark_results(self, results: List[BenchmarkResult], 
                              filename: str = "benchmark_results.json") -> str:
        """
        Save benchmark results to file.
        
        Args:
            results: List of benchmark results
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        # Convert results to serializable format
        results_data = []
        for result in results:
            result_dict = {
                'test_name': result.test_name,
                'duration_seconds': result.duration_seconds,
                'frames_processed': result.frames_processed,
                'avg_fps': result.avg_fps,
                'min_fps': result.min_fps,
                'max_fps': result.max_fps,
                'avg_frame_time_ms': result.avg_frame_time_ms,
                'avg_detection_time_ms': result.avg_detection_time_ms,
                'avg_memory_mb': result.avg_memory_mb,
                'peak_memory_mb': result.peak_memory_mb,
                'cpu_utilization_percent': result.cpu_utilization_percent,
                'success_rate': result.success_rate,
                'errors': result.errors,
                'timestamp': time.time()
            }
            results_data.append(result_dict)
        
        # Create benchmarks directory if it doesn't exist
        benchmark_dir = "benchmarks"
        os.makedirs(benchmark_dir, exist_ok=True)
        
        # Save results
        filepath = os.path.join(benchmark_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {filepath}")
        return filepath


# Global performance monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def initialize_performance_system(config=None) -> Tuple[PerformanceMonitor, PerformanceOptimizer]:
    """
    Initialize complete performance system.
    
    Args:
        config: System configuration
        
    Returns:
        Tuple of (monitor, optimizer)
    """
    monitor = get_performance_monitor()
    optimizer = PerformanceOptimizer(config)
    
    # Apply optimizations
    optimizer.apply_all_optimizations()
    
    return monitor, optimizer
