#!/usr/bin/env python3
"""
Basic Usage Example for Enhanced Blind Detection System

This script demonstrates the basic functionality of the Enhanced Blind Detection System
with step-by-step initialization and simple object detection.

Usage:
    python examples/basic_usage.py
    python examples/basic_usage.py --config custom_config.yaml
    python examples/basic_usage.py --duration 30
"""

import os
import sys
import time
import argparse
from typing import Optional

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

try:
    from src import main
    from src import config
    from src import logging_config
    from src import performance
    
    BlindDetectionApp = main.BlindDetectionApp
    ConfigManager = config.ConfigManager
    initialize_logging = logging_config.initialize_logging
    get_logger = logging_config.get_logger
    get_performance_monitor = performance.get_performance_monitor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running this script from the project root directory")
    print("and that all dependencies are installed.")
    sys.exit(1)


class BasicUsageExample:
    """
    Basic usage example demonstrating core functionality of the Enhanced Blind Detection System.
    """
    
    def __init__(self, config_path: Optional[str] = None, duration: float = 60.0):
        """
        Initialize the basic usage example.
        
        Args:
            config_path: Optional path to configuration file
            duration: How long to run the detection (seconds)
        """
        self.config_path = config_path or 'config.yaml'
        self.duration = duration
        self.app = None
        self.logger = None
        
    def setup_logging(self):
        """Setup logging for the example."""
        print("Setting up logging system...")
        try:
            logging_manager = initialize_logging()
            self.logger = get_logger(__name__)
            print("✓ Logging system initialized successfully")
        except Exception as e:
            print(f"✗ Failed to setup logging: {e}")
            return False
        return True
    
    def check_configuration(self):
        """Check and validate configuration."""
        print(f"Checking configuration file: {self.config_path}")
        
        if not os.path.exists(self.config_path):
            print(f"✗ Configuration file not found: {self.config_path}")
            print("  Creating default configuration...")
            self.create_default_config()
        
        try:
            config_manager = ConfigManager(self.config_path)
            config = config_manager.load_config()
            
            print("✓ Configuration loaded successfully")
            print(f"  Model: {config.model_name}")
            print(f"  Device: {config.device}")
            print(f"  Camera: Index {config.camera_index}")
            print(f"  Resolution: {config.frame_width}x{config.frame_height}")
            print(f"  Confidence threshold: {config.confidence_threshold}")
            print(f"  Target classes: {len(config.target_classes)} classes")
            
            return True
            
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            return False
    
    def create_default_config(self):
        """Create a default configuration file."""
        default_config = """# Basic Configuration for Enhanced Blind Detection System
# This is a simplified configuration for demonstration purposes

# Camera settings
camera_index: 0
frame_width: 640
frame_height: 480

# Model settings
model_name: "yolov8n.pt"
device: "cpu"  # Use "cuda" if you have a compatible GPU
confidence_threshold: 0.5

# Target object classes to detect
target_classes:
  - "person"
  - "car"
  - "bicycle"
  - "motorcycle"
  - "truck"
  - "chair"
  - "couch"
  - "bed"
  - "dining table"
  - "toilet"

# Audio settings
voice_rate: 200
voice_volume: 0.8
announcement_cooldown: 2.0

# Performance settings
enable_gpu: false
max_fps: 30
"""
        
        try:
            with open(self.config_path, 'w') as f:
                f.write(default_config)
            print(f"✓ Created default configuration: {self.config_path}")
        except Exception as e:
            print(f"✗ Failed to create default configuration: {e}")
    
    def initialize_application(self):
        """Initialize the main application."""
        print("Initializing Enhanced Blind Detection System...")
        
        try:
            self.app = BlindDetectionApp(self.config_path)
            print("✓ Application instance created")
            
            # Initialize all components
            if self.app.initialize():
                print("✓ All components initialized successfully")
                
                # Show system information
                self.show_system_info()
                return True
            else:
                print("✗ Component initialization failed")
                return False
                
        except Exception as e:
            print(f"✗ Application initialization failed: {e}")
            return False
    
    def show_system_info(self):
        """Display system information."""
        print("\nSystem Information:")
        
        try:
            # Get performance monitor for system info
            monitor = get_performance_monitor()
            system_info = monitor.system_info
            
            print(f"  CPU: {system_info.cpu_count} cores @ {system_info.cpu_freq_mhz:.0f}MHz")
            print(f"  Memory: {system_info.total_memory_gb:.1f}GB")
            
            if system_info.cuda_available:
                print(f"  GPU: {system_info.gpu_name} ({system_info.gpu_memory_gb:.1f}GB)")
            else:
                print("  GPU: Not available (using CPU)")
            
            if system_info.opencv_version:
                print(f"  OpenCV: {system_info.opencv_version}")
            
            print("")
            
        except Exception as e:
            print(f"  Could not retrieve system info: {e}")
    
    def run_detection_demo(self):
        """Run the main detection demonstration."""
        print(f"Starting object detection for {self.duration} seconds...")
        print("The system will:")
        print("  • Detect objects in camera view")
        print("  • Analyze spatial positioning")
        print("  • Provide audio feedback")
        print("  • Monitor performance metrics")
        print("\nPress Ctrl+C to stop early\n")
        
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        
        try:
            # Start the main detection loop
            self.app.running = True
            
            while self.app.running and (time.time() - start_time) < self.duration:
                try:
                    # Process one frame
                    success = self.app.process_frame()
                    
                    if success:
                        frame_count += 1
                        
                        # Check if we have performance metrics
                        monitor = get_performance_monitor()
                        current_metrics = monitor.get_current_metrics()
                        
                        if current_metrics.detection_count > detection_count:
                            detection_count = current_metrics.detection_count
                            elapsed = time.time() - start_time
                            print(f"[{elapsed:.1f}s] Detection #{detection_count} - "
                                  f"FPS: {current_metrics.fps:.1f}")
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.033)  # ~30 FPS max
                    
                except KeyboardInterrupt:
                    print("\nDetection stopped by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error during detection: {e}")
                    time.sleep(0.1)  # Brief pause before retrying
            
            # Show final statistics
            self.show_final_statistics(start_time, frame_count, detection_count)
            
        except Exception as e:
            print(f"Detection loop error: {e}")
            return False
        
        return True
    
    def show_final_statistics(self, start_time: float, frame_count: int, detection_count: int):
        """Show final detection statistics."""
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n{'='*50}")
        print("DETECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Duration: {total_time:.1f} seconds")
        print(f"Frames processed: {frame_count}")
        print(f"Objects detected: {detection_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        
        # Get performance summary
        try:
            monitor = get_performance_monitor()
            summary = monitor.get_performance_summary()
            
            if summary:
                fps_stats = summary.get('fps', {})
                memory_stats = summary.get('memory_mb', {})
                
                print(f"\nPerformance Metrics:")
                print(f"  FPS Range: {fps_stats.get('min', 0):.1f} - {fps_stats.get('max', 0):.1f}")
                print(f"  Memory Usage: {memory_stats.get('average', 0):.1f}MB average")
                print(f"  Peak Memory: {memory_stats.get('max', 0):.1f}MB")
                
        except Exception as e:
            print(f"Could not retrieve performance summary: {e}")
        
        print(f"{'='*50}")
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        
        try:
            if self.app:
                self.app.running = False
                self.app.cleanup()
            print("✓ Cleanup completed")
        except Exception as e:
            print(f"✗ Cleanup error: {e}")
    
    def run(self):
        """Run the complete basic usage demonstration."""
        print("Enhanced Blind Detection System - Basic Usage Example")
        print("=" * 60)
        
        try:
            # Step 1: Setup logging
            if not self.setup_logging():
                return False
            
            # Step 2: Check configuration
            if not self.check_configuration():
                return False
            
            # Step 3: Initialize application
            if not self.initialize_application():
                return False
            
            # Step 4: Run detection demo
            success = self.run_detection_demo()
            
            return success
            
        except KeyboardInterrupt:
            print("\nExample interrupted by user")
            return True
        except Exception as e:
            print(f"Example failed: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """Main function for the basic usage example."""
    parser = argparse.ArgumentParser(
        description="Basic usage example for Enhanced Blind Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/basic_usage.py                    # Run with default settings
  python examples/basic_usage.py --duration 30     # Run for 30 seconds
  python examples/basic_usage.py --config my.yaml  # Use custom config
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--duration', 
        type=float, 
        default=60.0,
        help='Duration to run detection in seconds (default: 60.0)'
    )
    
    parser.add_argument(
        '--create-config', 
        action='store_true',
        help='Create a default configuration file and exit'
    )
    
    args = parser.parse_args()
    
    # Handle config creation request
    if args.create_config:
        example = BasicUsageExample(args.config)
        example.create_default_config()
        print(f"Default configuration created: {args.config}")
        return 0
    
    # Run the basic usage example
    example = BasicUsageExample(args.config, args.duration)
    success = example.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
