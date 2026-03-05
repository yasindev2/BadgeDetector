# Installation Guide

## Prerequisites

- **Python 3.8+** (tested on 3.14.2)
- **Webcam or phone camera**
- **Wi-Fi network** (for phone access)

## Step-by-Step Installation

### 1. Clone or Download Repository

```bash
cd /path/to/project
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment (if not exists)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows
```

### 3. Install System Dependencies

**macOS:**
```bash
brew install cmake
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential
sudo apt-get install -y libboost-python-dev
```

**Windows:**
- Install Visual Studio Build Tools
- Install CMake from https://cmake.org/download/

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** This may take 5-10 minutes. The `face-recognition` package compiles dlib which takes time.

**Troubleshooting dlib installation:**
```bash
# If dlib fails, try installing separately:
pip install cmake
pip install dlib
pip install face-recognition
```

### 5. Download YOLOv8 Model (Automatic)

The YOLOv8n model will auto-download on first run (~6MB). Or manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 6. Verify Installation

```bash
python -c "import fastapi, cv2, face_recognition, ultralytics; print('✓ All dependencies installed')"
```

## Quick Test

### Test 1: Start Server

```bash
python server.py
```

You should see:
```
🚀 Staff ID Badge Detection System - Server Starting
Local access:  http://127.0.0.1:8000
Mobile access: http://192.168.x.x:8000
```

### Test 2: Health Check

Open another terminal:
```bash
curl http://localhost:8000
```

Expected response:
```json
{
  "status": "running",
  "device": "cpu",
  "half_precision": false,
  "badge_detector_loaded": false,
  "staff_db_loaded": false
}
```

### Test 3: Open Frontend

Open browser and go to `http://localhost:8000`

Click "Start Camera" and allow camera access.

You should see:
- Live camera feed
- Green boxes around detected persons
- Person count updating

## Optional: Enable Face Recognition

### 1. Prepare Staff Photos

```bash
mkdir staff_photos
```

Add photos with naming format: `firstname_lastname.jpg`
- Example: `john_doe.jpg`, `jane_smith.png`
- Requirements: Clear face, well-lit, front-facing

### 2. Build Database

```bash
python build_staff_db.py
```

Expected output:
```
Building Staff Database
Found 3 image(s) in staff_photos
...
✓ Database saved to: staff_db.pkl
Total staff members: 3
```

### 3. Test with Staff Recognition

Restart server and test with staff members in view.

## Optional: Enable Badge Detection

### 1. Collect Training Images

Collect 200+ images of people with ID badges:
- Various badge types
- Different angles and distances
- Different lighting conditions

### 2. Set Up Dataset

```bash
python train_badge_model.py --setup
```

### 3. Annotate Images

Use annotation tool (Roboflow recommended):
1. Sign up at https://roboflow.com
2. Create Object Detection project
3. Upload images
4. Draw boxes around badges
5. Export as YOLOv8 format
6. Extract to `badge_dataset/`

### 4. Train Model

```bash
python train_badge_model.py --epochs 100
```

Training takes 10-30 minutes depending on GPU/CPU.

### 5. Test Model

```bash
python train_badge_model.py --test
```

## GPU Acceleration (Optional)

For 3-5x faster processing:

### Check CUDA Availability

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Install CUDA PyTorch

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Restart server - should show:
```
Using device: cuda
Half precision (fp16): True
```

## Connecting from Mobile Phone

### 1. Find PC IP Address

**macOS:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Windows:**
```cmd
ipconfig
```

**Linux:**
```bash
hostname -I
```

### 2. Connect Phone to Same Wi-Fi

Ensure phone and PC are on the same network.

### 3. Open Browser on Phone

Navigate to: `http://<PC_IP>:8000`

Example: `http://192.168.1.100:8000`

### 4. Allow Camera Access

Click "Start Camera" and allow camera permissions.

## Troubleshooting

### Issue: "dlib won't install"

Solution:
```bash
# macOS
brew install cmake boost-python3

# Ubuntu
sudo apt-get install libboost-python-dev

# Windows - Install Visual Studio Build Tools
```

### Issue: "No module named 'cv2'"

Solution:
```bash
pip uninstall opencv-python
pip install opencv-python==4.10.0.84
```

### Issue: "Camera not accessible"

Solutions:
- Use HTTPS or localhost (browser security requirement)
- Check browser camera permissions
- Try different browser (Chrome recommended)
- For iOS Safari: Ensure not in Private Browsing mode

### Issue: "WebSocket connection failed"

Solutions:
- Check firewall allows port 8000
- Verify PC and phone on same network
- Try using IP address instead of hostname
- Disable VPN if active

### Issue: "Slow frame rate (<5 FPS)"

Solutions:
- Check GPU availability: `nvidia-smi` or look for "Using device: cuda" in logs
- Reduce frame rate: Edit `TARGET_FPS = 10` in `index.html`
- Increase frame skip threshold: Edit `FRAME_SKIP_THRESHOLD = 5` in `server.py`
- Close other applications using GPU/CPU

### Issue: "Face recognition not working"

Solutions:
- Ensure `staff_db.pkl` exists in root directory
- Check photo quality: clear face, well-lit, front-facing
- Verify person's face is in top 30% of detection box
- Try rebuilding database with better photos

### Issue: "Badge detection not accurate"

Solutions:
- Train with more images (200+ recommended)
- Include variety: different badges, angles, lighting
- Ensure consistent annotation (draw box around full badge + lanyard)
- Train for more epochs: `python train_badge_model.py --epochs 150`

## Support

For issues or questions, check:
1. README.md for general documentation
2. CLAUDE.md for architecture details
3. GitHub issues (if repository is public)

## Next Steps

After successful installation:
1. Add staff photos for face recognition
2. Collect and annotate badge images
3. Train badge detection model
4. Test full system with live camera
5. Deploy on dedicated machine for production use
