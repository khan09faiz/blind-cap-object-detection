#!/usr/bin/env python3
"""
Simple example runner that handles import paths correctly.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_basic_example():
    """Run basic usage example."""
    try:
        # Import directly from examples
        from examples import basic_usage
        
        # Create and run demo
        demo = basic_usage.BasicUsageDemo(duration=10.0)
        success = demo.run()
        return success
    except Exception as e:
        print(f"Error running basic example: {e}")
        return False

def run_advanced_example():
    """Run advanced demo example."""
    try:
        # Import directly from examples  
        from examples import advanced_demo
        
        # Create and run demo
        demo = advanced_demo.AdvancedDemo(scenario="indoor", duration=10.0)
        success = demo.run()
        return success
    except Exception as e:
        print(f"Error running advanced example: {e}")
        return False

def main():
    """Main example runner."""
    print("Enhanced Blind Detection System - Example Runner")
    print("=" * 50)
    
    print("\n1. Testing Basic Usage Example...")
    basic_success = run_basic_example()
    
    print(f"\n2. Testing Advanced Demo Example...")
    advanced_success = run_advanced_example()
    
    print("\n" + "=" * 50)
    print("EXAMPLE RUNNER RESULTS")
    print("=" * 50)
    print(f"Basic Example: {'✅ SUCCESS' if basic_success else '❌ FAILED'}")
    print(f"Advanced Example: {'✅ SUCCESS' if advanced_success else '❌ FAILED'}")
    
    overall_success = basic_success and advanced_success
    print(f"\nOverall: {'✅ ALL EXAMPLES WORKING' if overall_success else '❌ SOME EXAMPLES FAILED'}")
    
    return 0 if overall_success else 1

if __name__ == '__main__':
    sys.exit(main())
