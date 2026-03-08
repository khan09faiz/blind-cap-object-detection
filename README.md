<div align="center">

# 🦯 Blind-Cap — AI Assistive Vision System

**Real-time object detection, depth estimation, and intelligent audio guidance for visually impaired users.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![YOLOv10](https://img.shields.io/badge/Model-YOLOv10-orange)](https://github.com/THU-MIG/yolov10)
[![MiDaS](https://img.shields.io/badge/Depth-MiDaS__small-green)](https://github.com/isl-org/MiDaS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

![Blind-Cap Demo](Screenshot%202025-07-23%20133559.png)

</div>

---

## Table of Contents

1. [What Is This?](#what-is-this)
2. [How It Works](#how-it-works)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Configuration & Scenarios](#configuration--scenarios)
6. [Training Your Own Model](#training-your-own-model)
7. [Detected Object Classes](#detected-object-classes)
8. [Priority Speech System](#priority-speech-system)
9. [Roadmap & Future Vision](#roadmap--future-vision)
10. [Contributing](#contributing)
11. [Collaborators](#collaborators)

---

## What Is This?

**Blind-Cap** is an open-source assistive vision system built for visually impaired people. It attaches to a standard webcam (or wearable camera) and continuously:

- **Detects** objects in the live camera frame using a custom-trained YOLOv10 model
- **Estimates depth/distance** to each object using the MiDaS monocular depth network — no special depth sensor needed
- **Announces objects aloud** via offline text-to-speech, telling the user *what* the object is, *where* it is (left / ahead / right), and *how far* away it is
- **Prioritises warnings** — fast-moving hazards like cars and buses are announced before benign objects like chairs or benches

The entire pipeline runs **on CPU** with no internet connection required after first setup, making it viable on low-cost hardware.

---

## How It Works

```
Webcam frame
     │
     ▼
┌─────────────────┐     every frame
│  YOLOv10 Detect │──────────────────► bounding boxes + class labels + confidence
└─────────────────┘
     │
     ▼
┌─────────────────┐     every 3 frames (configurable)
│  MiDaS Depth   │──────────────────► per-pixel relative depth map
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  IoU Tracker    │──────────────────► stable track IDs, suppresses re-announcements
└─────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│         Priority Sorter              │
│  Tier-1 (cars/buses)  score 0–9.99   │
│  Tier-2 (poles/steps) score 10–19.99 │
│  Safe objects         score 20–29.99 │
└──────────────────────────────────────┘
     │
     ▼
┌─────────────────┐
│  pyttsx3 TTS    │──────────────────► "Warning! Car on left, 2 metres"
└─────────────────┘
```

**Direction logic** divides the frame into thirds:
- Left third → "on the left"
- Centre third → "ahead"
- Right third → "on the right"

**Distance** is derived from the median depth value inside the bounding box, converted to a relative metre estimate using MiDaS output scale.

---

## Architecture

```
blind-cap-object-detection/
├── main.py                   # Entry point — opens webcam, runs pipeline loop
├── run_app.py                # Convenience launcher (wraps main.py, extra arg handling)
├── launch.py                 # Auto-detecting launcher (picks scenario from camera caps)
├── config.yaml               # All configuration (5 scenarios)
├── requirements.txt
│
├── core/                     # Foundational utilities
│   ├── config.py             # ConfigManager — loads + deep-merges YAML scenarios
│   ├── error_handling.py     # Centralised exception handling
│   ├── frame_processor.py    # Frame pre-processing helpers
│   ├── logging_config.py     # Structured logging setup
│   ├── navigation.py         # Navigation state machine
│   └── performance.py        # FPS + resource monitoring
│
├── models/
│   ├── detector.py           # YOLOv10 wrapper (auto-detects best.pt vs default)
│   ├── depth_estimator.py    # MiDaS_small wrapper + bbox→distance helper
│   └── best.pt               # Custom-trained weights (15 Open Images classes)
│
├── tracking/
│   └── tracker.py            # IoU tracker with per-track announcement cooldown
│
├── audio/
│   └── speech.py             # Non-blocking pyttsx3 TTS (daemon thread + queue)
│
├── pipeline/
│   └── vision_pipeline.py    # Orchestrator: detect → track → depth → sort → speak
│
├── utils/
│   ├── direction.py          # Left/ahead/right classifier
│   └── obstacle_rules.py     # Danger tiers + priority_score()
│
└── training/
    ├── colab_training.ipynb  # Full Colab training notebook (Open Images + YOLOv10)
    └── dataset.yaml          # Class definitions for training
```

---

## Quick Start

### Prerequisites

- Python 3.10 or newer
- A webcam (built-in or USB)
- ~2 GB free disk space (model weights + MiDaS cache)

### 1. Clone & Install

```bash
git clone https://github.com/aviral-rai1875/blind-cap-object-detection.git
cd blind-cap-object-detection
pip install -r requirements.txt
```

> **GPU users:** swap `torch` for the CUDA wheel matching your driver:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Add the Custom Model Weights

Place `best.pt` (trained weights) into the `models/` folder.  
*(If you don't have `best.pt` yet, see [Training Your Own Model](#training-your-own-model).  
The system falls back to `yolov10n.pt` which auto-downloads from Ultralytics.)*

### 3. Run

```bash
# Custom-trained model (recommended)
python run_app.py --scenario custom

# Standard webcam scenarios
python run_app.py --scenario indoor
python run_app.py --scenario outdoor
python run_app.py --scenario default

# Auto-detecting launcher (picks scenario based on camera resolution)
python launch.py
```

### 4. Camera Window Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `z` | Toggle zone lines |
| `f` | Toggle FPS display |
| `ESC` | Quit |

---

## Configuration & Scenarios

All settings live in [`config.yaml`](config.yaml). Five scenarios ship out of the box:

| Scenario | Model | Use Case |
|---|---|---|
| `default` | yolov10n.pt | General purpose |
| `indoor` | yolov10s.pt | Home / office navigation |
| `outdoor` | yolov10s.pt | Street / park navigation |
| `high_performance` | yolov10m.pt | When GPU is available |
| `custom` | models/best.pt | **15-class Open Images trained model** |

Key tunable parameters:

```yaml
detector:
  confidence_threshold: 0.4     # raise for fewer false positives

depth:
  frame_interval: 3             # run depth every N frames (lower = more accurate, slower)

tracker:
  announcement_cooldown_frames: 60   # frames before re-announcing the same object

speech:
  rate: 150                     # words per minute
  volume: 1.0
```

---

## Training Your Own Model

We train on a subset of the [Google Open Images](https://storage.googleapis.com/openimages/web/index.html) dataset using **Google Colab's free GPU**.

### Steps

1. Open [`training/colab_training.ipynb`](training/colab_training.ipynb) in [Google Colab](https://colab.research.google.com/)
2. Set Runtime → **T4 GPU**
3. Run all 11 cells — the notebook will:
   - Install `oidv6` and `ultralytics`
   - Download selected classes from Open Images
   - Convert labels to YOLO format
   - Train `yolov10n` for 50 epochs
   - Export `best.pt`
   - Download `best.pt` to your machine
4. Copy `best.pt` to `models/best.pt`
5. Run: `python run_app.py --scenario custom`

### Current Trained Classes (15)

`person` · `chair` · `table` · `couch` · `bed` · `door` · `bench` · `waste_container` · `fire_hydrant` · `postbox` · `car` · `bus` · `truck` · `bicycle` · `dog`

---

## Detected Object Classes

### Default / COCO Classes (yolov10n.pt)

80 standard COCO classes including people, vehicles, furniture, animals, and common household items.

### Custom Classes (best.pt — Open Images)

| Class | Danger Tier | Why It Matters |
|---|---|---|
| car | 🔴 Tier-1 | Fast-moving, high-impact hazard |
| bus | 🔴 Tier-1 | Fast-moving, high-impact hazard |
| truck | 🔴 Tier-1 | Fast-moving, high-impact hazard |
| bicycle | 🟡 Tier-2 | Moving obstacle |
| waste_container | 🟡 Tier-2 | Stationary obstacle at path level |
| fire_hydrant | 🟡 Tier-2 | Low stationary obstacle |
| person | ✅ Safe | Navigable with guidance |
| chair | ✅ Safe | Common indoor obstacle |
| door | ✅ Safe | Navigation landmark |
| bench | ✅ Safe | Rest point |
| couch | ✅ Safe | Indoor furniture |
| bed | ✅ Safe | Indoor furniture |
| table | ✅ Safe | Indoor furniture |
| postbox | ✅ Safe | Navigation landmark |
| dog | ✅ Safe | Moving animal |

---

## Priority Speech System

When multiple objects are in frame, the system speaks them in **danger-first order**:

```
Score 0–9.99   → Tier-1 (car, bus, truck, motorcycle) — closest first
Score 10–19.99 → Tier-2 (bicycle, pole, fire_hydrant, waste_container)
Score 20–29.99 → Safe objects (person, chair, bench, dog …)
```

Example output for a busy street scene:
```
"Warning! Car on the left, 2 metres"
"Warning! Bus ahead, 5 metres"
"Person on the right, 3 metres"
"Bench ahead, 1 metre"
```

---

## Roadmap & Future Vision

This project is actively evolving. Below are the planned features and integrations we want to build next:

### 🕶️ Meta Ray-Ban Smart Glasses Integration

Connect Blind-Cap to **Meta Ray-Ban glasses** as a wearable assistive device.

- **Streaming Mode** — glasses camera streams live video over Wi-Fi/USB; the pipeline processes it on a paired phone or laptop and speaks through the glasses' built-in speakers
- **Blind Person Mode** — fully hands-free: no screen, purely audio guidance through the glasses earphones
- **Navigation Mode** — turn-by-turn street navigation combined with object detection
- **Social Mode** — recognise faces and read text (OCR) aloud for social interactions

### 🤟 Sign Language Interpreter

Add a dedicated sign language module:

- Real-time hand pose detection using **MediaPipe Hands**
- ASL / BSL gesture classifier (CNN or LSTM over 21 keypoints)
- Two-way bridge: sign language → spoken audio AND spoken audio → on-screen text subtitles
- Useful for both the visually impaired user *and* their deaf/mute conversation partners

### 🧠 Expanded Object Detection

- Add 50+ new classes: traffic signs, stairs, escalators, ATMs, crosswalk signals, shop entrances, bus stops, elevators
- Night vision / low-light enhancement preprocessing
- Crowd density estimation ("very crowded ahead")
- Vehicle speed estimation from consecutive frames

### 🗺️ Indoor Navigation Map

- Build real-time occupancy grids from depth maps
- Room-scale SLAM (Simultaneous Localisation and Mapping)
- Remembered landmarks ("you passed the kitchen door")

### 📱 Mobile App

- Android / iOS app wrapping the pipeline via ONNX Runtime Mobile
- Real-time inference on-device (no server required)
- Vibration haptic feedback as a secondary warning channel

### ☁️ Cloud Companion Mode

- Optional cloud mode: stream video to server, get richer AI responses back
- GPT-4o Vision integration for scene description ("describe what is around me")
- Emergency SOS: if the user falls (detected via sudden camera shake), send alert to emergency contact

### 🎛️ Adaptive Verbosity

- Learn user preferences: "I already know my office layout, skip furniture"
- Quiet mode: only speak about new/moving objects
- Re-announce only when distance changes significantly (not every frame)

---

## Contributing

Contributions are very welcome! Here's how to get started:

```bash
# Fork the repo, clone your fork
git clone https://github.com/<your-username>/blind-cap-object-detection.git
cd blind-cap-object-detection

# Create a feature branch
git checkout -b feature/sign-language-module

# Make your changes, then run the test suite
python -m pytest tests/ -v         # unit tests (if present)
python -c "import pipeline.vision_pipeline"   # quick smoke test

# Push and open a Pull Request
git push origin feature/sign-language-module
```

### Areas That Need Help

| Area | Skills Needed |
|---|---|
| Meta Ray-Ban integration | Python, network streaming, BLE/Wi-Fi |
| Sign language model | MediaPipe, PyTorch, gesture datasets |
| Mobile app | Flutter or React Native + ONNX |
| Expanded class training | Data labelling, Colab, YOLO fine-tuning |
| Indoor SLAM | OpenCV, point clouds, robotics |
| Testing & CI | pytest, GitHub Actions |

Please open an **Issue** before starting large features so we can coordinate.

---

## Collaborators

| Name | GitHub |
|---|---|
| Aviral Rai | [@aviral-rai1875](https://github.com/aviral-rai1875) |
| Mohammad Faiz Khan | [@khan09faizi](https://github.com/khan09faizi) |

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ to make the world more accessible.
</div>

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
