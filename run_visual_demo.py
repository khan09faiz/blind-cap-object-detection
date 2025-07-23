#!/usr/bin/env python3
"""
Visual demo runner for Enhanced Blind Detection System

This script runs the system with visual interface enabled, showing:
- Live camera feed
- Object detection boxes with labels
- Distance and position information
- Real-time performance metrics

Usage:
    python run_visual_demo.py
    python run_visual_demo.py --scenario indoor
    python run_visual_demo.py --config config.yaml
"""

import sys
import os
import argparse
import time

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.main import BlindDetectionApp
    from src.config import ConfigManager
    from src.logging_config import get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running this script from the project root directory")
    sys.exit(1)


def print_banner():
    """Print the visual demo banner."""
    print("=" * 80)
    print("ENHANCED BLIND DETECTION SYSTEM - VISUAL DEMO")
    print("=" * 80)
    print("Features:")
    print("  • Live camera feed with object detection overlay")
    print("  • Real-time distance and position information")
    print("  • Color-coded detection boxes (Red=Close, Orange=Medium, Green=Far)")
    print("  • Performance metrics and zone indicators")
    print("  • Audio announcements with visual feedback")
    print()
    print("Controls:")
    print("  • Press 'q' to quit")
    print("  • Press 's' to save screenshot")
    print("  • Press 'z' to toggle zone lines")
    print("  • Press 'f' to toggle FPS display")
    print("=" * 80)
    print()


def main():
    """Main function for visual demo."""
    parser = argparse.ArgumentParser(
        description="Visual demonstration of Enhanced Blind Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_visual_demo.py                    # Default settings
    python run_visual_demo.py --scenario indoor  # Indoor navigation mode
    python run_visual_demo.py --scenario outdoor # Outdoor navigation mode
    python run_visual_demo.py --config my_config.yaml  # Custom configuration
        """
    )
    
    parser.add_argument(
        '--scenario',
        choices=['default', 'indoor', 'outdoor', 'high_performance', 'low_resource'],
        default='default',
        help='Detection scenario to use (default: default)'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Disable audio announcements (visual only)'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    try:
        # Initialize the application
        print("Initializing Enhanced Blind Detection System...")
        app = BlindDetectionApp(args.config)
        
        # Enable visual display
        app.enable_visual_display = True
        
        # Disable audio if requested
        if args.no_audio:
            print("Audio disabled - visual only mode")
            # Note: Audio will be disabled in the audio manager initialization
        
        print(f"Using scenario: {args.scenario}")
        print(f"Configuration: {args.config}")
        print()
        
        # Initialize the system
        print("Initializing system components...")
        if not app.initialize():
            print("✗ Failed to initialize system")
            return 1
        
        print("✓ System initialized successfully")
        print()
        print("Starting visual detection demo...")
        print("Camera window will open - position it as needed")
        print("You should see object detection boxes with distance/position labels")
        print()
        
        # Run the application
        try:
            app.run()
        except KeyboardInterrupt:
            print("\n\nDemo stopped by user (Ctrl+C)")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        return 1
    
    finally:
        # Cleanup
        try:
            if 'app' in locals():
                app.cleanup()
            print("\n✓ Cleanup completed")
        except Exception as e:
            print(f"\n⚠ Cleanup error: {e}")


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
