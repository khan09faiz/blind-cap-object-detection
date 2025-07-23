# Enhanced Blind Detection System

A real-time object detection system designed to assist visually impaired users with navigation through audio announcements and visual feedback.

## Features

🎯 **Real-time Object Detection** - Uses YOLOv8 for accurate object detection
👁️ **Visual Interface** - Live camera feed with detection boxes and labels  
🔊 **Audio Announcements** - Intelligent text-to-speech guidance
📍 **Spatial Analysis** - Distance and position information
⚡ **GPU Acceleration** - CUDA support for better performance
🎛️ **Multiple Scenarios** - Indoor, outdoor, and performance-optimized modes

# Colaborator
**Aviral Rai 
**Mohammad Faiz Khan 

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Visual Camera System**
   ```bash
   # Start visual detection with camera popup window
   python run_visual_demo.py --scenario indoor
   
   # Alternative scenarios
   python run_visual_demo.py --scenario outdoor
   python run_visual_demo.py --scenario high_performance
   ```

3. **Camera Window Controls**
   - **'q'** - Quit and close camera window
   - **'s'** - Save screenshot of current view
   - **'z'** - Toggle zone lines (shows detection areas)
   - **'f'** - Toggle FPS display (shows performance)

   > 💡 **Tip**: Press 's' while the camera is running to save a screenshot to the project folder

## What You'll See

When you run the system, a **camera window will popup** showing:
- ✅ **Live camera feed** from your webcam
- 🟢 **Green boxes** around far objects 
- 🟡 **Orange boxes** around medium distance objects
- 🔴 **Red boxes** around close objects (warning!)
- 📝 **Labels** showing object name and distance
- 🎯 **Crosshair** in center for navigation reference
- 📊 **Performance metrics** (if enabled with 'f' key)

### Screenshot Example

![Enhanced Blind Detection System in Action](Screenshot%202025-07-23%20133559.png)

*Example showing the camera interface with object detection boxes, distance labels, and visual indicators*

## System Requirements

- Python 3.8+
- **Webcam or camera device** (required for visual interface)
- Windows/Linux/macOS
- Optional: NVIDIA GPU with CUDA for better performance

## Audio Feedback

The system provides **intelligent audio announcements**:
- 🔴 **Urgent warnings** for objects directly ahead and close
- 🟡 **Caution alerts** for objects to the side or medium distance  
- 🟢 **Information** about far objects and general navigation
- 🎯 **Spatial guidance** with left/center/right positioning

## Configuration

The system uses `config.yaml` with built-in scenarios:
- **Indoor** - Optimized for indoor navigation
- **Outdoor** - Better for outdoor environments  
- **High Performance** - Uses more resources for better accuracy
- **Low Resource** - Lighter processing for older computers

## Usage Examples

```bash
# Most common usage - indoor camera view
python run_visual_demo.py --scenario indoor

# For outdoor use
python run_visual_demo.py --scenario outdoor

# High performance (if you have a good computer/GPU)
python run_visual_demo.py --scenario high_performance

# Audio only mode (no camera window)
python run_app.py --audio-only
```

## Project Structure

```
blind-cap-object-detection/
├── src/                          # Core application modules
│   ├── main.py                   # Main application controller
│   ├── detector.py               # YOLOv8 object detection
│   ├── visual_interface.py       # Camera window and visual overlay
│   ├── audio.py                  # Text-to-speech announcements
│   ├── spatial.py                # Distance and position analysis
│   └── frame_processor.py        # Camera handling
├── run_visual_demo.py            # ⭐ Main script to run camera system
├── run_app.py                    # Alternative runner (audio-only mode)
├── config.yaml                   # Configuration settings
├── yolov8n.pt                    # Pre-trained AI model
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details
