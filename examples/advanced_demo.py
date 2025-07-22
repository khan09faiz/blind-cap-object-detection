#!/usr/bin/env python3
"""
Advanced Usage Demonstration for Enhanced Blind Detection System

This script provides a comprehensive demonstration of advanced features including
configuration customization, performance monitoring, and different detection scenarios.

Usage:
    python examples/advanced_demo.py
    python examples/advanced_demo.py --scenario indoor
    python examples/advanced_demo.py --scenario outdoor
    python examples/advanced_demo.py --scenario performance
"""

import os
import sys
import time
import argparse
from typing import Optional, Dict, Any
import threading
import signal

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

try:
    from src import main
    from src import config
    from src import logging_config
    from src import performance
    from src import audio
    from src import spatial
    
    BlindDetectionApp = main.BlindDetectionApp
    ConfigManager = config.ConfigManager
    initialize_logging = logging_config.initialize_logging
    get_logger = logging_config.get_logger
    get_performance_monitor = performance.get_performance_monitor
    PerformanceBenchmark = performance.PerformanceBenchmark
    AudioManager = audio.AudioManager
    SpatialAnalyzer = spatial.SpatialAnalyzer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running this script from the project root directory")
    sys.exit(1)


class AdvancedDemo:
    """
    Advanced demonstration showcasing comprehensive features of the Enhanced Blind Detection System.
    """
    
    def __init__(self, scenario: str = "indoor", duration: float = 120.0):
        """
        Initialize the advanced demonstration.
        
        Args:
            scenario: Detection scenario (indoor, outdoor, performance)
            duration: How long to run the detection (seconds)
        """
        self.scenario = scenario
        self.duration = duration
        self.app = None
        self.logger = None
        self.running = False
        self.performance_monitor = None
        
        # Statistics tracking
        self.stats = {
            'frames_processed': 0,
            'objects_detected': 0,
            'announcements_made': 0,
            'start_time': None,
            'scenario_events': []
        }
    
    def print_banner(self):
        """Print the demonstration banner."""
        print("=" * 80)
        print("ENHANCED BLIND DETECTION SYSTEM - ADVANCED DEMONSTRATION")
        print("=" * 80)
        print(f"Scenario: {self.scenario.title()}")
        print(f"Duration: {self.duration} seconds")
        print(f"Features: Performance monitoring, advanced audio, spatial analysis")
        print("=" * 80)
        print()
    
    def setup_scenario_configuration(self) -> str:
        """Setup configuration based on the selected scenario."""
        config_map = {
            'indoor': 'examples/config_indoor.yaml',
            'outdoor': 'examples/config_outdoor.yaml', 
            'performance': 'examples/config_high_performance.yaml',
            'low_resource': 'examples/config_low_resource.yaml'
        }
        
        config_path = config_map.get(self.scenario, 'config.yaml')
        
        # Create config if it doesn't exist
        if not os.path.exists(config_path):
            print(f"Configuration file not found: {config_path}")
            print("Using default configuration...")
            config_path = 'config.yaml'
            
            if not os.path.exists(config_path):
                print("Creating default configuration...")
                self.create_default_config(config_path)
        
        print(f"Using configuration: {config_path}")
        return config_path
    
    def create_default_config(self, config_path: str):
        """Create a default configuration for the demo."""
        default_config = f"""# Advanced Demo Configuration - {self.scenario.title()} Scenario
# Auto-generated for demonstration purposes

camera_index: 0
frame_width: 640
frame_height: 480

model_name: "yolov8n.pt"
device: "cpu"
confidence_threshold: 0.5

target_classes:
  - "person"
  - "car"
  - "bicycle"
  - "chair"
  - "couch"
  - "dining table"
  - "laptop"
  - "cell phone"
  - "book"
  - "bottle"

voice_rate: 200
voice_volume: 0.8
announcement_cooldown: 2.0

# Performance monitoring enabled for demo
performance:
  enable_monitoring: true
  report_interval: 10.0
  
logging:
  level: "INFO"
  file_logging: true
  console_logging: true
"""
        
        try:
            with open(config_path, 'w') as f:
                f.write(default_config)
            print(f"✓ Created configuration: {config_path}")
        except Exception as e:
            print(f"✗ Failed to create configuration: {e}")
    
    def setup_logging_and_monitoring(self, config_path: str):
        """Setup logging and performance monitoring."""
        print("Setting up advanced monitoring...")
        
        try:
            # Initialize logging
            logging_manager = initialize_logging()
            self.logger = get_logger(__name__)
            print("✓ Logging system initialized")
            
            # Initialize performance monitoring
            self.performance_monitor = get_performance_monitor()
            print("✓ Performance monitoring initialized")
            
            # Create application instance
            self.app = BlindDetectionApp(config_path)
            print("✓ Application instance created")
            
            return True
            
        except Exception as e:
            print(f"✗ Setup failed: {e}")
            return False
    
    def demonstrate_configuration_features(self):
        """Demonstrate advanced configuration features."""
        print("\n" + "-" * 60)
        print("CONFIGURATION FEATURES DEMONSTRATION")
        print("-" * 60)
        
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            print("Configuration Analysis:")
            print(f"  Model: {config.model_name}")
            print(f"  Device: {config.device}")
            print(f"  Resolution: {config.frame_width}x{config.frame_height}")
            print(f"  Target Classes: {len(config.target_classes)} objects")
            print(f"  Confidence Threshold: {config.confidence_threshold}")
            print(f"  Audio Settings: Rate={config.voice_rate}, Volume={config.voice_volume}")
            
            # Demonstrate configuration validation
            is_valid = config_manager.validate_config()
            print(f"  Configuration Valid: {'✓' if is_valid else '✗'}")
            
            # Show scenario-specific optimizations
            self.show_scenario_optimizations(config)
            
        except Exception as e:
            print(f"Configuration demonstration error: {e}")
    
    def show_scenario_optimizations(self, config):
        """Show optimizations specific to the current scenario."""
        print(f"\n{self.scenario.title()} Scenario Optimizations:")
        
        if self.scenario == 'indoor':
            print("  • Higher confidence threshold for indoor accuracy")
            print("  • Focus on furniture and people detection") 
            print("  • Shorter announcement cooldown for frequent obstacles")
            print("  • Lower resolution for balanced performance")
            
        elif self.scenario == 'outdoor':
            print("  • Vehicle and traffic detection prioritized")
            print("  • Higher resolution for outdoor visibility")
            print("  • Increased volume for outdoor environments")
            print("  • Extended danger object warnings")
            
        elif self.scenario == 'performance':
            print("  • Maximum resolution and model size")
            print("  • GPU acceleration enabled")
            print("  • Comprehensive object detection")
            print("  • Advanced performance monitoring")
    
    def demonstrate_audio_features(self):
        """Demonstrate advanced audio management features."""
        print("\n" + "-" * 60)
        print("AUDIO FEATURES DEMONSTRATION")
        print("-" * 60)
        
        try:
            # Create audio manager for demonstration
            audio_manager = AudioManager()
            
            print("Audio System Capabilities:")
            print(f"  TTS Engine: {'Available' if audio_manager.tts_engine else 'Not Available'}")
            print(f"  Voice Rate: {audio_manager.voice_rate}")
            print(f"  Voice Volume: {audio_manager.voice_volume}")
            print(f"  Announcement Cooldown: {audio_manager.announcement_cooldown}s")
            
            # Demonstrate different message types
            print("\nMessage Types:")
            sample_messages = [
                ("Detection", "Person detected in center zone"),
                ("Navigation", "Clear path ahead"),
                ("Warning", "Vehicle approaching from left"),
                ("Status", "System ready for navigation")
            ]
            
            for msg_type, message in sample_messages:
                print(f"  {msg_type}: \"{message}\"")
            
            # Demonstrate spatial audio descriptions
            print("\nSpatial Descriptions:")
            spatial_examples = [
                ("Left Zone", "Object on your left side"),
                ("Center Zone", "Object directly ahead"),
                ("Right Zone", "Object on your right side"),
                ("Near Distance", "Close object - use caution"),
                ("Far Distance", "Distant object detected")
            ]
            
            for location, description in spatial_examples:
                print(f"  {location}: \"{description}\"")
                
        except Exception as e:
            print(f"Audio demonstration error: {e}")
    
    def demonstrate_spatial_analysis(self):
        """Demonstrate spatial analysis capabilities."""
        print("\n" + "-" * 60)
        print("SPATIAL ANALYSIS DEMONSTRATION")
        print("-" * 60)
        
        try:
            # Create spatial analyzer
            spatial = SpatialAnalyzer(frame_width=640, frame_height=480)
            
            print("Spatial Analysis Configuration:")
            print(f"  Frame Dimensions: {spatial.frame_width}x{spatial.frame_height}")
            print(f"  Zone Boundaries: Left={spatial.left_boundary}, Right={spatial.right_boundary}")
            print(f"  Distance Thresholds: Near={spatial.near_threshold}, Far={spatial.far_threshold}")
            
            # Demonstrate zone detection with example bounding boxes
            print("\nZone Detection Examples:")
            example_boxes = [
                ([50, 100, 150, 300], "Left zone object"),
                ([250, 100, 350, 300], "Center zone object"),
                ([450, 100, 550, 300], "Right zone object"),
                ([100, 50, 200, 150], "Small distant object"),
                ([100, 200, 500, 450], "Large near object")
            ]
            
            for bbox, description in example_boxes:
                try:
                    position = spatial.analyze_position(bbox)
                    zone = position['zone']
                    distance = position['distance_category']
                    print(f"  {description}: {zone} zone, {distance} distance")
                except Exception as e:
                    print(f"  {description}: Analysis error - {e}")
                    
        except Exception as e:
            print(f"Spatial analysis demonstration error: {e}")
    
    def run_detection_with_monitoring(self):
        """Run detection with comprehensive performance monitoring."""
        print("\n" + "-" * 60)
        print("DETECTION WITH PERFORMANCE MONITORING")
        print("-" * 60)
        
        print(f"Starting {self.scenario} detection scenario...")
        print("Features active:")
        print("  • Real-time object detection")
        print("  • Spatial analysis and positioning")
        print("  • Intelligent audio feedback")
        print("  • Performance monitoring")
        print("  • Error handling and recovery")
        print()
        
        # Initialize application
        if not self.app.initialize():
            print("✗ Failed to initialize application")
            return False
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start performance monitoring thread
        monitor_thread = threading.Thread(target=self.performance_monitoring_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            while self.running and (time.time() - self.stats['start_time']) < self.duration:
                try:
                    # Process frame with monitoring
                    success = self.app.process_frame()
                    
                    if success:
                        self.stats['frames_processed'] += 1
                        
                        # Check for new detections
                        current_metrics = self.performance_monitor.get_current_metrics()
                        if current_metrics.detection_count > self.stats['objects_detected']:
                            self.stats['objects_detected'] = current_metrics.detection_count
                            
                            # Log detection event
                            elapsed = time.time() - self.stats['start_time']
                            event = f"[{elapsed:.1f}s] Detection #{self.stats['objects_detected']}"
                            self.stats['scenario_events'].append(event)
                            
                            # Show real-time progress
                            if self.stats['objects_detected'] % 5 == 0:
                                print(f"Progress: {self.stats['objects_detected']} objects detected, "
                                      f"{current_metrics.fps:.1f} FPS")
                    
                    # Scenario-specific behavior
                    self.handle_scenario_specific_events()
                    
                    # Control frame rate
                    time.sleep(0.033)  # ~30 FPS max
                    
                except KeyboardInterrupt:
                    print("\n\nDetection stopped by user")
                    break
                except Exception as e:
                    self.logger.error(f"Detection error: {e}")
                    time.sleep(0.1)
        
        finally:
            self.running = False
        
        return True
    
    def handle_scenario_specific_events(self):
        """Handle events specific to the current scenario."""
        elapsed = time.time() - self.stats['start_time']
        
        # Simulate scenario-specific events
        if self.scenario == 'indoor' and elapsed > 30 and len(self.stats['scenario_events']) < 3:
            self.stats['scenario_events'].append(f"[{elapsed:.1f}s] Indoor navigation mode active")
            
        elif self.scenario == 'outdoor' and elapsed > 20 and len(self.stats['scenario_events']) < 5:
            self.stats['scenario_events'].append(f"[{elapsed:.1f}s] Outdoor safety monitoring active")
            
        elif self.scenario == 'performance' and elapsed > 15:
            # Performance scenario generates more events
            if len(self.stats['scenario_events']) % 10 == 0:
                metrics = self.performance_monitor.get_current_metrics()
                event = f"[{elapsed:.1f}s] Performance check: {metrics.fps:.1f} FPS, {metrics.memory_mb:.1f}MB"
                self.stats['scenario_events'].append(event)
    
    def performance_monitoring_loop(self):
        """Background performance monitoring loop."""
        last_report = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Report every 20 seconds
                if current_time - last_report >= 20.0:
                    metrics = self.performance_monitor.get_current_metrics()
                    summary = self.performance_monitor.get_performance_summary()
                    
                    print(f"\nPerformance Update:")
                    print(f"  FPS: {metrics.fps:.1f} (avg: {summary.get('fps', {}).get('average', 0):.1f})")
                    print(f"  Memory: {metrics.memory_mb:.1f}MB")
                    print(f"  Frame Time: {metrics.frame_time_ms:.1f}ms")
                    print(f"  Detection Time: {metrics.detection_time_ms:.1f}ms")
                    
                    last_report = current_time
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(1.0)
    
    def show_final_results(self):
        """Show comprehensive final results."""
        total_time = time.time() - self.stats['start_time']
        
        print("\n" + "=" * 80)
        print("ADVANCED DEMONSTRATION RESULTS")
        print("=" * 80)
        
        # Basic statistics
        print(f"Scenario: {self.scenario.title()}")
        print(f"Duration: {total_time:.1f} seconds")
        print(f"Frames Processed: {self.stats['frames_processed']}")
        print(f"Objects Detected: {self.stats['objects_detected']}")
        print(f"Average FPS: {self.stats['frames_processed'] / total_time:.1f}")
        
        # Performance metrics
        try:
            summary = self.performance_monitor.get_performance_summary()
            fps_stats = summary.get('fps', {})
            memory_stats = summary.get('memory_mb', {})
            
            print(f"\nPerformance Metrics:")
            print(f"  FPS Range: {fps_stats.get('min', 0):.1f} - {fps_stats.get('max', 0):.1f}")
            print(f"  Average Memory: {memory_stats.get('average', 0):.1f}MB")
            print(f"  Peak Memory: {memory_stats.get('max', 0):.1f}MB")
            print(f"  Total Detections: {summary.get('total_detections', 0)}")
            print(f"  System Uptime: {summary.get('uptime_seconds', 0):.1f}s")
            
        except Exception as e:
            print(f"Could not retrieve performance summary: {e}")
        
        # Scenario events
        if self.stats['scenario_events']:
            print(f"\nScenario Events ({len(self.stats['scenario_events'])}):")
            for event in self.stats['scenario_events'][-10:]:  # Show last 10 events
                print(f"  {event}")
            if len(self.stats['scenario_events']) > 10:
                print(f"  ... and {len(self.stats['scenario_events']) - 10} more events")
        
        # System information
        try:
            system_info = self.performance_monitor.system_info
            print(f"\nSystem Information:")
            print(f"  CPU: {system_info.cpu_count} cores @ {system_info.cpu_freq_mhz:.0f}MHz")
            print(f"  Memory: {system_info.total_memory_gb:.1f}GB")
            if system_info.cuda_available:
                print(f"  GPU: {system_info.gpu_name}")
            if system_info.opencv_version:
                print(f"  OpenCV: {system_info.opencv_version}")
                
        except Exception as e:
            print(f"Could not retrieve system information: {e}")
        
        print("=" * 80)
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up demonstration resources...")
        
        try:
            self.running = False
            if self.app:
                self.app.running = False
                self.app.cleanup()
            print("✓ Cleanup completed")
        except Exception as e:
            print(f"✗ Cleanup error: {e}")
    
    def run(self):
        """Run the complete advanced demonstration."""
        try:
            # Setup signal handler for graceful shutdown
            signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))
            
            self.print_banner()
            
            # Setup configuration
            config_path = self.setup_scenario_configuration()
            
            # Setup logging and monitoring
            if not self.setup_logging_and_monitoring(config_path):
                return False
            
            # Feature demonstrations
            self.demonstrate_configuration_features()
            self.demonstrate_audio_features()
            self.demonstrate_spatial_analysis()
            
            # Main detection demonstration
            success = self.run_detection_with_monitoring()
            
            # Show results
            if success:
                self.show_final_results()
            
            return success
            
        except KeyboardInterrupt:
            print("\n\nAdvanced demonstration interrupted by user")
            return True
        except Exception as e:
            print(f"\nAdvanced demonstration failed: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """Main function for the advanced demonstration."""
    parser = argparse.ArgumentParser(
        description="Advanced demonstration of Enhanced Blind Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  indoor       - Indoor navigation with furniture and people detection
  outdoor      - Outdoor navigation with vehicle and traffic detection  
  performance  - High-performance mode with comprehensive monitoring
  low_resource - Low-resource mode for constrained systems

Examples:
  python examples/advanced_demo.py --scenario indoor
  python examples/advanced_demo.py --scenario outdoor --duration 90
  python examples/advanced_demo.py --scenario performance
        """
    )
    
    parser.add_argument(
        '--scenario', 
        choices=['indoor', 'outdoor', 'performance', 'low_resource'],
        default='indoor',
        help='Detection scenario to demonstrate (default: indoor)'
    )
    
    parser.add_argument(
        '--duration', 
        type=float, 
        default=120.0,
        help='Duration to run demonstration in seconds (default: 120.0)'
    )
    
    args = parser.parse_args()
    
    # Run the advanced demonstration
    demo = AdvancedDemo(args.scenario, args.duration)
    success = demo.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
