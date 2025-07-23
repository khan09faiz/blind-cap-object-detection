#!/usr/bin/env python3
"""
Optimized launcher for Enhanced Blind Detection System
Automatically detects best settings for your hardware
"""

import sys
import os
import torch
import cv2
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def detect_optimal_settings():
    """Detect optimal settings based on hardware."""
    settings = {
        'scenario': 'default',
        'model': 'yolov8n.pt',
        'resolution': (640, 480),
        'device': 'cpu'
    }
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory > 4:  # 4GB+ GPU
            settings['scenario'] = 'high_performance'
            settings['model'] = 'yolov8m.pt'
            settings['device'] = 'cuda'
        else:
            settings['scenario'] = 'default'
            settings['device'] = 'cuda'
    
    # Check camera resolution
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width >= 1280 and height >= 720:
            settings['resolution'] = (1280, 720)
        cap.release()
    
    return settings

def main():
    parser = argparse.ArgumentParser(description="Enhanced Blind Detection System - Optimized Launcher")
    parser.add_argument('--visual', action='store_true', help='Enable visual interface')
    parser.add_argument('--audio-only', action='store_true', help='Audio only mode')
    parser.add_argument('--scenario', choices=['auto', 'indoor', 'outdoor', 'high_performance'], 
                       default='auto', help='Detection scenario')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Blind Detection System - Optimized Launcher")
    print("=" * 60)
    
    # Detect optimal settings
    if args.scenario == 'auto':
        settings = detect_optimal_settings()
        print(f"Auto-detected settings: {settings}")
    else:
        settings = {'scenario': args.scenario}
    
    # Import and run application
    try:
        from src.main import BlindDetectionApp
        
        app = BlindDetectionApp('config.yaml')
        
        # Configure based on arguments
        if args.visual:
            app.enable_visual_display = True
            print("✅ Visual interface enabled")
        
        if args.audio_only:
            app.enable_visual_display = False
            print("🔊 Audio-only mode enabled")
        
        print("\nInitializing system...")
        if app.initialize():
            print("✅ System ready!")
            print("\nStarting detection...")
            app.run()
        else:
            print("❌ Initialization failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
