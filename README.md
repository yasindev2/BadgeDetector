# Real-Time Staff ID Badge Detection System

A real-time computer vision system that detects persons, recognizes staff members, and verifies ID badge compliance through a phone camera in the browser.

## Features

- **Real-time WebSocket streaming** - Continuous frame processing without HTTP overhead
- **Multi-stage detection pipeline**:
  - Person detection using YOLOv8n
  - Badge detection on chest region
  - Face recognition for staff identification
- **Violation detection** - Automatic flagging when staff members don't wear badges
- **Mobile-optimized** - Works on phone cameras via browser (Chrome/Safari)
- **GPU acceleration** - Automatic fp16 inference when CUDA available
- **Smart frame skipping** - Drops old frames when processing is busy

## System Architecture

```
Phone Camera → WebSocket → Backend Queue → Detection Pipeline → Annotated Frame
                                              ↓
                                         YOLOv8 Person
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                            Badge Detection      Face Recognition
                            (Chest Region)       (Head Region)
                                    ↓                   ↓
                                    └─────────┬─────────┘
                                              ↓
                                      Violation Check
```

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Installing `face-recognition` and `dlib` may require additional system libraries:
- **macOS**: `brew install cmake`
- **Ubuntu**: `sudo apt-get install cmake libboost-python-dev`

### 2. Test Person Detection (Immediate)

You can test the system immediately with just person detection:

```bash
# Start server
python server.py
```

The server will display:
```
🚀 Staff ID Badge Detection System - Server Starting
Local access:  http://127.0.0.1:8000
Mobile access: http://192.168.x.x:8000
```

Open the mobile access URL on your phone's browser and click "Start Camera".

### 3. Add Staff Database (Optional)

To enable face recognition:

```bash
# Create staff photos directory
mkdir staff_photos

# Add photos (format: firstname_lastname.jpg)
# Example: john_doe.jpg, jane_smith.png

# Build database
python build_staff_db.py
```

**With roles:**
```bash
# Create roles file
python build_staff_db.py --create-roles

# Edit staff_roles.json to assign roles
# {
#   "john_doe": "Security Guard",
#   "jane_smith": "Manager"
# }

# Build database
python build_staff_db.py
```

### 4. Train Badge Detector (Optional)

To enable badge detection:

```bash
# Create dataset structure
python train_badge_model.py --setup

# Add annotated images to badge_dataset/
# See badge_dataset/README.md for annotation guide

# Train model
python train_badge_model.py

# Test model
python train_badge_model.py --test
```

## Usage

### Starting the Server

```bash
python server.py
```

### Connecting from Mobile

1. Ensure phone and PC are on same Wi-Fi network
2. Open browser on phone (Chrome or Safari)
3. Navigate to `http://<PC_IP>:8000`
4. Click "Start Camera"
5. Allow camera access when prompted

### Server Endpoints

- `GET /` - Health check and status
- `WebSocket /ws` - Real-time frame streaming endpoint

## Performance Tuning

### Target Performance
- **Frame rate**: 15 FPS capture, 5-10 FPS processing
- **Latency**: <200ms per frame (GPU), <500ms (CPU)

### GPU Acceleration

With CUDA-enabled GPU:
- Automatic fp16 inference
- 3-5x faster processing
- Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### CPU Optimization

For CPU-only systems:
- Reduce frame rate: Edit `TARGET_FPS` in `index.html`
- Use smaller model: Replace `yolov8n.pt` with `yolov8n-pose.pt`
- Increase `FRAME_SKIP_THRESHOLD` in `server.py`

## File Structure

```
Ryden/
├── server.py                  # FastAPI WebSocket backend
├── index.html                 # Mobile-optimized frontend
├── build_staff_db.py          # Staff database builder
├── train_badge_model.py       # Badge model training script
├── requirements.txt           # Python dependencies
├── staff_db.pkl              # Generated staff database
├── staff_photos/             # Staff photos for face recognition
├── badge_dataset/            # Badge training dataset
└── models/
    └── badge_detector.pt     # Trained badge detection model
```

## Configuration

### Server Configuration (server.py)

```python
MAX_QUEUE_SIZE = 2              # Frame queue size
TARGET_FPS = 15                 # Target frame rate
FRAME_SKIP_THRESHOLD = 3        # Drop frames threshold
```

### Client Configuration (index.html)

```javascript
const TARGET_FPS = 15;          // Frame capture rate
const CANVAS_WIDTH = 640;       // Frame width
const CANVAS_HEIGHT = 480;      // Frame height
```

## Troubleshooting

### Camera Not Working
- Ensure HTTPS or localhost (browser security requirement)
- Check camera permissions in browser settings
- Try different browser (Chrome recommended)

### Connection Failed
- Verify PC and phone on same network
- Check firewall settings (allow port 8000)
- Try local IP address instead of hostname

### Slow Processing
- Check if GPU is detected: Look for "Using device: cuda" in logs
- Reduce frame rate in frontend
- Close other GPU-intensive applications

### Face Recognition Not Working
- Ensure `staff_db.pkl` exists in project root
- Check photo quality (clear, well-lit faces)
- Verify face is visible in head region (top 30% of person bbox)

### Badge Detection Not Working
- Ensure `models/badge_detector.pt` exists
- Check if badge is in chest region (15%-55% of person height)
- Verify model was trained with enough data (200+ images recommended)

## Development

### Testing Without Phone

Use webcam for testing:
```python
# Modify index.html line 191:
facingMode: 'user'  # Instead of 'environment'
```

### Adding Custom Annotations

Edit `process_frame()` in `server.py` to add custom visualizations.

### Adjusting Detection Regions

```python
# Chest region (badge): server.py line 120
chest_y1 = y1 + int(person_height * 0.15)  # Top: 15%
chest_y2 = y1 + int(person_height * 0.55)  # Bottom: 55%

# Head region (face): server.py line 156
head_y2 = y1 + int(person_height * 0.3)    # Top 30%
```

## Security Considerations

- **Local network only** - Do not expose to public internet
- **Staff photos** - Store securely, include consent
- **HTTPS** - Use reverse proxy (nginx) for production
- **Authentication** - Add authentication layer for production use

## License

This project is for educational and internal use. Ensure compliance with privacy laws when deploying with face recognition.

## Credits

- **YOLOv8** by Ultralytics
- **face_recognition** by Adam Geitgey
- **FastAPI** by Sebastián Ramírez
# BadgeDetector
