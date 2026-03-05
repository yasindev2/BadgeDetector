# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Real-Time Staff ID Badge Detection System** - A computer vision system using phone cameras to detect persons, recognize staff via face recognition, and verify ID badge compliance in real-time. Built with FastAPI WebSocket backend and mobile-optimized HTML frontend.

## Core Architecture

### Detection Pipeline (server.py)

The system uses a multi-stage detection pipeline:

1. **Person Detection** (`detect_persons()`) - YOLOv8n detects all persons in frame (COCO class 0)
2. **Badge Detection** (`detect_badge_on_chest()`) - Custom YOLO checks chest region (15%-55% of person height) for hanging badge
3. **Face Recognition** (`recognize_face()`) - face_recognition library identifies staff from head region (top 30%) against pickle database
4. **Violation Flagging** - If staff detected without badge, flag as VIOLATION

### WebSocket Communication

- **Real-time streaming** via WebSocket (not HTTP polling)
- **Background thread queue** with smart frame skipping (drops oldest when busy, keeps newest)
- **Target performance**: <200ms per frame processing
- **Client-Server Protocol**:
  - Client sends: Binary JPEG frames at 15fps
  - Server returns: JSON with base64 JPEG + stats

```json
{
  "frame": "base64_encoded_jpeg",
  "stats": {
    "persons_count": 2,
    "staff_count": 1,
    "violations_count": 1,
    "violations": [{"person_id": 0, "name": "John Doe", "reason": "No badge detected"}],
    "process_time_ms": 145.2,
    "fps": 6.9
  }
}
```

### Smart Frame Queue (server.py:312-342)

The background async task processes frames asynchronously:
- Max 2 frames in queue (`MAX_QUEUE_SIZE`)
- If queue > `FRAME_SKIP_THRESHOLD`, drop oldest frame
- Always process newest frame
- Prevents backlog and maintains real-time performance

### Device Acceleration

- **GPU Auto-detection**: Uses CUDA with fp16 for 3-5x speedup
- **CPU Fallback**: Gracefully falls back with full precision
- Models loaded on startup with `load_models()`

## Running the Application

### Quick Start (Person Detection Only)

```bash
# Start server
./start_server.sh
# OR
python server.py
```

Access from phone: `http://<PC_IP>:8000` (or `https://` if certificates present)

### Full System with Face Recognition

```bash
# 1. Build staff database
mkdir staff_photos
# Add photos: firstname_lastname.jpg
python build_staff_db.py

# 2. Start server
python server.py
```

### With Badge Detection

```bash
# 1. Set up dataset structure
python train_badge_model.py --setup

# 2. Annotate images in badge_dataset/ (use Roboflow/LabelImg)

# 3. Train model
python train_badge_model.py --epochs 100 --batch 16

# 4. Start server (auto-detects trained model)
python server.py
```

## Key Commands

### Server Operations
- Start server: `python server.py` or `./start_server.sh`
- Server listens on: `0.0.0.0:8000`
- Health check: `curl http://localhost:8000/status`
- WebSocket endpoint: `ws://localhost:8000/ws`

### Database Management
- Build staff DB: `python build_staff_db.py`
- Create roles template: `python build_staff_db.py --create-roles`
- Staff DB location: `staff_db.pkl` (root directory)

### Model Training
- Setup dataset: `python train_badge_model.py --setup`
- Train badge model: `python train_badge_model.py --epochs 100 --batch 16`
- Test trained model: `python train_badge_model.py --test`
- Model location: `models/badge_detector.pt`

### Testing
- Run system tests: `python test_system.py`
- Tests imports, GPU, staff DB, badge model, webcam, network

## File Structure

```
server.py              # FastAPI WebSocket backend (main server)
index.html             # Mobile frontend with camera streaming
build_staff_db.py      # Utility: Build face recognition database
train_badge_model.py   # Utility: Train custom badge detector
test_system.py         # System verification tests
requirements.txt       # Python dependencies
start_server.sh        # Server startup script
staff_db.pkl           # Generated: Staff face encodings (pickle)
staff_photos/          # Input: Staff photos for face recognition
badge_dataset/         # Input: Annotated images for badge training
models/
  badge_detector.pt    # Generated: Trained badge detection model
```

## Key Implementation Details

### Frame Processing Flow (server.py:215-309)

`process_frame()` orchestrates the pipeline:
1. Decode JPEG to numpy array
2. Call `detect_persons()` - returns list of person bboxes
3. For each person:
   - Call `detect_badge_on_chest()` on chest crop (15%-55% height)
   - Call `recognize_face()` on head region (top 30% height)
4. Check violations (staff without badge)
5. Annotate frame with colored bboxes and labels
6. Return encoded JPEG + stats JSON

### Detection Regions

**Badge (chest region):**
```python
# server.py:128-129
chest_y1 = y1 + int(person_height * 0.15)  # Top: 15%
chest_y2 = y1 + int(person_height * 0.55)  # Bottom: 55%
```

**Face (head region):**
```python
# server.py:169 - Uses full person region for face detection
# More reliable than just head crop
face_locations = face_recognition.face_locations(rgb_person, model="hog")
```

### Staff Database Format (staff_db.pkl)

```python
{
    "encodings": [array, array, ...],  # 128-dim face encodings
    "names": ["John Doe", "Jane Smith", ...],
    "roles": ["Security Guard", "Manager", ...]
}
```

Built by `build_staff_db.py` from `staff_photos/` directory.

### Badge Dataset Structure (YOLO format)

```
badge_dataset/
  data.yaml           # Dataset config
  images/
    train/            # Training images
    val/              # Validation images
  labels/
    train/            # YOLO format labels (.txt)
    val/
```

YOLO label format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)

## Configuration Tuning

### Performance Adjustments (server.py:34-37)

```python
MAX_QUEUE_SIZE = 2              # Frame buffer size
TARGET_FPS = 15                 # Target frame rate
FRAME_SKIP_THRESHOLD = 3        # Drop frames when queue > this
```

### Frontend Settings (index.html:235-238)

```javascript
const TARGET_FPS = 15;          // Capture rate
const CANVAS_WIDTH = 640;       // Frame resolution
const CANVAS_HEIGHT = 480;
```

### Model Paths (server.py:40-42)

```python
STAFF_DB_PATH = Path("staff_db.pkl")
BADGE_MODEL_PATH = Path("models/badge_detector.pt")
PERSON_MODEL_PATH = "yolov8n.pt"  # Auto-downloads if not present
```

## Development Notes

### Virtual Environment
- Python 3.14.2 (note: very recent, may have library compatibility issues)
- Activate: `source .venv/bin/activate`
- Install: `pip install -r requirements.txt`

### Dependencies (requirements.txt)
- FastAPI + Uvicorn + WebSockets for real-time server
- YOLOv8 (ultralytics) for person and badge detection
- face_recognition + dlib for staff identification
- OpenCV for image processing
- PyTorch for GPU acceleration

### Graceful Degradation

The system works in three modes:
1. **Person detection only** - Works out of the box with YOLOv8n
2. **+ Face recognition** - Requires `staff_db.pkl`
3. **+ Badge detection** - Requires `models/badge_detector.pt`

Each component is optional; system adapts to what's available.

### HTTPS/SSL Support

- If `cert.pem` and `key.pem` exist, server runs in HTTPS mode
- HTTPS required for camera access on mobile (not localhost)
- Generate self-signed cert: `openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365`

### Mobile Access Requirements

- PC and phone must be on same Wi-Fi network
- Browser requires HTTPS or localhost for camera access
- Chrome/Safari mobile recommended
- Accept self-signed certificate warning

## Important Notes

- **Python 3.14** - Very recent version, some libraries may have compatibility issues
- **GPU Acceleration** - Auto-detected, significantly improves performance (3-5x faster)
- **Staff Database** - System works without `staff_db.pkl` (no face recognition)
- **Badge Model** - System works without `badge_detector.pt` (no badge detection)
- **Security** - Local network only; add authentication for production use
- **Privacy** - Store staff photos securely, obtain consent for face recognition
