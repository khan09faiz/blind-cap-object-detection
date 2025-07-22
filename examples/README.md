# Enhanced Blind Detection System - Examples and Usage Guide

This directory contains comprehensive examples and usage demonstrations for the Enhanced Blind Detection System. These examples help users understand how to configure, customize, and use the system for different scenarios.

## 📁 Example Files Overview

### 🚀 Usage Examples

#### `basic_usage.py`
**Complete beginner-friendly demonstration**
- Step-by-step system initialization
- Configuration checking and validation
- System information display
- Basic object detection demonstration
- Performance monitoring
- Error handling and cleanup

```bash
# Run basic usage example
python examples/basic_usage.py

# With custom configuration
python examples/basic_usage.py --config examples/config_indoor.yaml

# Shorter demonstration
python examples/basic_usage.py --duration 60
```

#### `advanced_demo.py`
**Comprehensive feature demonstration**
- Scenario-specific configurations
- Advanced performance monitoring
- Audio system capabilities
- Spatial analysis features
- Background processing
- Statistics tracking

```bash
# Run advanced demo for indoor navigation
python examples/advanced_demo.py --scenario indoor

# Run outdoor scenario
python examples/advanced_demo.py --scenario outdoor

# High-performance demonstration
python examples/advanced_demo.py --scenario performance --duration 180
```

### ⚙️ Configuration Examples

#### `config_indoor.yaml`
**Optimized for indoor navigation (homes, offices)**
- Furniture and people detection focus
- Moderate resolution (640x480) for balanced performance
- Indoor-specific object classes
- Shorter announcement cooldown for frequent obstacles
- Reduced voice volume for indoor environments

**Key Features:**
- Target objects: person, chair, couch, dining table, laptop, book, bottle
- Confidence threshold: 0.6 (higher accuracy)
- Frame rate: 30 FPS
- Announcement cooldown: 1.5 seconds

#### `config_outdoor.yaml`
**Optimized for outdoor navigation (streets, parks)**
- Vehicle and traffic detection priority
- Higher resolution (1280x720) for distant objects
- Safety-focused settings
- Enhanced danger detection
- Increased voice volume for outdoor noise

**Key Features:**
- Target objects: person, car, truck, bicycle, motorcycle, traffic light, stop sign
- Confidence threshold: 0.4 (more sensitive for safety)
- Frame rate: 25 FPS
- Emergency announcement settings
- Extended detection range

#### `config_high_performance.yaml`
**Maximum performance for powerful systems**
- GPU acceleration enabled
- Highest resolution (1920x1080)
- Comprehensive object detection (all 80 COCO classes)
- Advanced features enabled
- Detailed performance monitoring

**Key Features:**
- Device: CUDA (GPU acceleration)
- Model: YOLOv8l.pt (large model)
- All available object classes
- Real-time performance analytics
- Advanced spatial analysis

#### `config_low_resource.yaml`
**Optimized for constrained systems (Raspberry Pi, older computers)**
- Minimal resource usage
- Lower resolution (320x240)
- Essential objects only
- Reduced processing load
- Optimized for efficiency

**Key Features:**
- Device: CPU only
- Model: YOLOv8n.pt (nano model)
- Essential objects: person, car, bicycle, chair
- Frame rate: 15 FPS
- Minimal logging

## 🎯 Usage Scenarios

### Scenario 1: First-Time Setup
```bash
# 1. Verify installation
python verify_installation.py

# 2. Run basic usage example
python examples/basic_usage.py

# 3. Test different configurations
python examples/basic_usage.py --config examples/config_indoor.yaml
```

### Scenario 2: Indoor Navigation
```bash
# Optimized for home/office navigation
python examples/advanced_demo.py --scenario indoor --duration 300

# Use indoor configuration directly
python -m src.main --config examples/config_indoor.yaml
```

### Scenario 3: Outdoor Navigation
```bash
# Optimized for street/park navigation
python examples/advanced_demo.py --scenario outdoor --duration 600

# Use outdoor configuration directly
python -m src.main --config examples/config_outdoor.yaml
```

### Scenario 4: Performance Testing
```bash
# Maximum performance demonstration
python examples/advanced_demo.py --scenario performance --duration 120

# Use high-performance configuration
python -m src.main --config examples/config_high_performance.yaml
```

### Scenario 5: Resource-Constrained Systems
```bash
# Minimal resource usage
python -m src.main --config examples/config_low_resource.yaml

# Test with low-resource demo
python examples/basic_usage.py --config examples/config_low_resource.yaml --duration 60
```

## 🛠️ Customization Guide

### Creating Custom Configurations

1. **Copy a base configuration:**
```bash
cp examples/config_indoor.yaml my_custom_config.yaml
```

2. **Edit key parameters:**
```yaml
# Camera settings
camera_index: 0          # Change camera if needed
frame_width: 640         # Adjust resolution
frame_height: 480

# Detection settings
confidence_threshold: 0.5 # Lower = more sensitive
target_classes:          # Add/remove objects
  - "person"
  - "car"
  - "bicycle"

# Audio settings
voice_rate: 200          # Speaking speed
voice_volume: 0.8        # Volume level
announcement_cooldown: 2.0 # Delay between announcements
```

3. **Test your configuration:**
```bash
python examples/basic_usage.py --config my_custom_config.yaml
```

### Configuration Parameters Guide

#### 🎥 Camera Settings
- `camera_index`: Camera device index (usually 0)
- `frame_width`: Video width in pixels
- `frame_height`: Video height in pixels
- `fps`: Target frames per second

#### 🎯 Detection Settings
- `model_name`: YOLOv8 model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `device`: Processing device ("cpu", "cuda", "mps")
- `confidence_threshold`: Minimum detection confidence (0.0-1.0)
- `target_classes`: List of objects to detect

#### 🔊 Audio Settings
- `voice_rate`: Speech rate (words per minute)
- `voice_volume`: Volume level (0.0-1.0)
- `announcement_cooldown`: Minimum time between announcements
- `voice_id`: Specific voice to use (optional)

#### 📊 Performance Settings
- `enable_monitoring`: Enable performance tracking
- `report_interval`: Performance report frequency
- `max_fps`: Maximum processing frame rate
- `memory_limit_mb`: Memory usage limit

## 🧪 Testing and Validation

### Running Example Tests
```bash
# Test example scripts
python -m pytest tests/test_examples.py -v

# Test specific example
python -m pytest tests/test_examples.py::TestBasicUsageExample -v

# Test configurations
python -m pytest tests/test_examples.py::TestExampleConfigurations -v
```

### Performance Benchmarking
```bash
# Run performance benchmark
python examples/advanced_demo.py --scenario performance --duration 300

# Compare configurations
python scripts/benchmark_configs.py
```

### Manual Testing Checklist

1. **Configuration Validation:**
   - [ ] Configuration loads without errors
   - [ ] All required fields present
   - [ ] Valid parameter values
   - [ ] Target classes exist in model

2. **Hardware Compatibility:**
   - [ ] Camera access working
   - [ ] Audio output functioning
   - [ ] GPU acceleration (if available)
   - [ ] Adequate memory available

3. **Detection Accuracy:**
   - [ ] Objects detected correctly
   - [ ] Appropriate confidence levels
   - [ ] Spatial positioning accurate
   - [ ] Audio announcements clear

4. **Performance Metrics:**
   - [ ] Frame rate meets requirements
   - [ ] Memory usage within limits
   - [ ] CPU usage reasonable
   - [ ] No memory leaks

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Camera Not Working
```bash
# Check camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Try different camera index
# Edit configuration: camera_index: 1
```

#### Audio Not Working
```bash
# Test audio system
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('test'); engine.runAndWait()"

# Install audio dependencies (Linux)
sudo apt-get install espeak espeak-data
```

#### Low Performance
```bash
# Use low-resource configuration
python -m src.main --config examples/config_low_resource.yaml

# Check system resources
python examples/advanced_demo.py --scenario performance --duration 60
```

#### Detection Issues
```bash
# Lower confidence threshold
# Edit configuration: confidence_threshold: 0.3

# Use larger model (if system can handle it)
# Edit configuration: model_name: "yolov8m.pt"
```

## 📈 Advanced Usage

### Custom Object Classes
To detect specific objects, modify the `target_classes` in your configuration:

```yaml
target_classes:
  - "person"      # Always include for safety
  - "car"         # Vehicles
  - "bicycle"     # Bikes
  - "chair"       # Furniture
  - "laptop"      # Electronics
  - "cell phone"  # Small objects
  - "book"        # Reading materials
  - "bottle"      # Containers
```

### Performance Optimization
For maximum performance:

1. **Use GPU acceleration:**
```yaml
device: "cuda"
model_name: "yolov8l.pt"
```

2. **Optimize resolution:**
```yaml
frame_width: 1280
frame_height: 720
```

3. **Enable performance monitoring:**
```yaml
performance:
  enable_monitoring: true
  report_interval: 5.0
```

### Multi-Camera Setup
For multiple cameras, create separate configurations:

```bash
# Camera 1 (main)
python -m src.main --config config_camera1.yaml &

# Camera 2 (side)
python -m src.main --config config_camera2.yaml &
```

## 📚 Additional Resources

- **Main Documentation:** `../README.md`
- **API Reference:** `../docs/api_reference.md`
- **Configuration Guide:** `../docs/configuration.md`
- **Troubleshooting:** `../docs/troubleshooting.md`
- **Performance Tuning:** `../docs/performance.md`

## 🤝 Contributing

To add new examples:

1. Create your example script in `examples/`
2. Add corresponding configuration in `examples/`
3. Write tests in `tests/test_examples.py`
4. Update this documentation
5. Submit a pull request

## 📄 License

This example collection is part of the Enhanced Blind Detection System and follows the same license terms.

---

*For questions or support, please refer to the main project documentation or create an issue in the project repository.*
