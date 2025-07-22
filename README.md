# Enhanced Blind Detection System

An AI-powered object detection system designed to assist visually impaired individuals with real-time navigation and obstacle detection using computer vision and audio feedback.

## 🌟 Features

- **Real-time Object Detection** - Uses YOLOv8 for accurate object detection
- **Spatial Analysis** - Determines object positions (left, center, right zones)
- **Audio Feedback** - Natural language announcements via text-to-speech
- **Intelligent Navigation** - Context-aware guidance and safety warnings
- **Performance Optimized** - Efficient processing for real-time operation
- **Configurable** - Multiple configuration options for different scenarios

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Audio output capability (speakers/headphones)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/khan09faiz/blind-cap-object-detection.git
   cd blind-cap-object-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python run_app.py
   ```

## 📖 Usage

### Basic Usage

Run with default settings:
```bash
python run_app.py
```

Run with specific scenario:
```bash
# Indoor navigation
python run_app.py --scenario indoor

# Outdoor navigation  
python run_app.py --scenario outdoor

# High performance mode
python run_app.py --scenario high_performance

# Low resource mode
python run_app.py --scenario low_resource
```

### Example Scripts

Run demonstration examples:
```bash
# Basic usage demonstration
python examples/basic_usage.py

# Advanced feature demonstration with scenarios
python examples/advanced_demo.py --scenario indoor
python examples/advanced_demo.py --scenario outdoor
```

## ⚙️ Configuration

The system uses a unified YAML configuration file with multiple scenarios. Choose your scenario based on your use case:

### Available Scenarios

```bash
# Default balanced settings
python run_app.py --scenario default

# Indoor navigation (home/office)
python run_app.py --scenario indoor

# Outdoor navigation (street/public spaces)
python run_app.py --scenario outdoor

# High performance (powerful hardware)
python run_app.py --scenario high_performance

# Low resource (older/slower hardware)
python run_app.py --scenario low_resource
```

### Configuration Structure

Each scenario includes these settings:

```yaml
scenario_name:
  # Model settings
  model_name: "yolov8n.pt"        # Model size (n/s/m/l/x)
  confidence_threshold: 0.5       # Detection confidence
  device: "auto"                  # Processing device
  
  # Camera settings
  camera_index: 0                 # Camera device
  frame_width: 640               # Resolution width
  frame_height: 480              # Resolution height
  
  # Audio settings
  voice_rate: 200                # Speech speed
  voice_volume: 0.8              # Volume level
  announcement_cooldown: 2.0     # Delay between announcements
  
  # Object classes to detect
  target_classes:
    - "person"
    - "car"
    - "chair"
    # ... more objects
```

### Custom Configuration

You can create your own configuration by copying and modifying any scenario in `config.yaml`.

## 🏗️ Project Structure

```
blind-cap-object-detection/
├── src/                     # Source code
│   ├── main.py             # Main application
│   ├── detector.py         # Object detection
│   ├── spatial.py          # Spatial analysis
│   ├── audio.py            # Audio management
│   ├── navigation.py       # Navigation guidance
│   ├── frame_processor.py  # Camera handling
│   ├── config.py           # Configuration management
│   ├── error_handling.py   # Error handling
│   └── logging_config.py   # Logging setup
├── examples/               # Usage examples
│   ├── basic_usage.py      # Basic demonstration
│   └── advanced_demo.py    # Advanced features
├── config.yaml            # Default configuration
├── requirements.txt       # Python dependencies
├── run_app.py             # Application runner
└── README.md              # This file
```

## 🔧 System Requirements

### Minimum Requirements
- **CPU:** Dual-core processor
- **Memory:** 4GB RAM
- **Python:** 3.8+
- **Camera:** Any USB webcam
- **OS:** Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements
- **CPU:** Quad-core processor
- **Memory:** 8GB RAM
- **GPU:** CUDA-compatible GPU (optional, for better performance)
- **Camera:** HD webcam (1080p)

## 🎯 Object Detection Classes

The system can detect and announce the following objects:

**People & Living**
- Person

**Vehicles**
- Car, Truck, Bus
- Bicycle, Motorcycle

**Furniture**
- Chair, Couch, Bed
- Dining table, Desk

**Electronics**
- Laptop, TV, Cell phone

**Household Items**
- Bottle, Cup, Book
- And many more...

## 🛠️ Troubleshooting

### Common Issues

**Camera not working:**
- Check camera permissions
- Ensure no other applications are using the camera
- Try different camera index in configuration

**Audio not working:**
- Check system audio settings
- Verify text-to-speech engine is installed
- Adjust volume settings in configuration

**Low performance:**
- Close other resource-intensive applications
- Consider upgrading hardware
- Adjust detection settings in configuration

### Getting Help

1. Check the console output for error messages
2. Review configuration settings
3. Open an issue on GitHub with detailed error information

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Techstack

- **YOLOv8** by Ultralytics for object detection
- **OpenCV** for computer vision capabilities
- **pyttsx3** for text-to-speech functionality
- **PyTorch** for deep learning framework


