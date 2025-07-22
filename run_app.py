#!/usr/bin/env python3
"""
Module runner for the Enhanced Blind Detection System.
This script properly handles imports and runs the main application.
"""

import os
import sys
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

def run_application(config_path=None):
    """Run the main blind detection application."""
    try:
        # Import main application components
        from src import main, config
        
        # Initialize configuration
        if config_path:
            config_manager = config.ConfigManager(config_path)
        else:
            config_manager = config.ConfigManager()
        
        # Create and run application
        app = main.BlindDetectionApp(config_manager.config_path)
        
        print("Starting Enhanced Blind Detection System...")
        
        if app.initialize():
            print("✅ System initialized successfully")
            app.run()
        else:
            print("❌ Failed to initialize system")
            return False
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        return True
    except Exception as e:
        print(f"Application error: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Blind Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                                    # Use default config
  python run_app.py --config config.yaml              # Use specific config
  python run_app.py --config examples/config_indoor.yaml  # Use indoor config
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    success = run_application(args.config)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
