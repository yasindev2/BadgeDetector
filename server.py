"""
Real-Time Staff ID Badge Detection System - WebSocket Backend
FastAPI server with YOLOv8 person detection, face recognition, and badge detection
"""

import asyncio
import base64
import io
import json
import os
import pickle
import time
from collections import deque
from pathlib import Path
from threading import Thread, Lock
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from ultralytics import YOLO

# Optional: face_recognition (can be installed later)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  face_recognition not installed - face recognition disabled")

# Configuration
MAX_QUEUE_SIZE = 2  # Keep only latest frames
TARGET_FPS = 15
FRAME_SKIP_THRESHOLD = 3  # Drop frames if queue exceeds this

# Paths
STAFF_DB_PATH = Path("staff_db.pkl")
BADGE_MODEL_PATH = Path("models/badge_detector.pt")
PERSON_MODEL_PATH = "yolov8n.pt"  # Will auto-download if not present

# Global state
app = FastAPI()
frame_queue = deque(maxlen=MAX_QUEUE_SIZE)
queue_lock = Lock()
processing = False

# Models (loaded on startup)
person_detector: Optional[YOLO] = None
badge_detector: Optional[YOLO] = None
staff_database: Optional[Dict] = None

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
use_half = device == "cuda"  # fp16 only on GPU

print(f"Using device: {device}")
print(f"Half precision (fp16): {use_half}")


def load_models():
    """Load YOLO models and staff database on startup"""
    global person_detector, badge_detector, staff_database

    print("Loading YOLOv8 person detector...")
    person_detector = YOLO(PERSON_MODEL_PATH)
    person_detector.to(device)
    if use_half:
        person_detector.model.half()

    # Load custom badge detector if available
    if BADGE_MODEL_PATH.exists():
        print("Loading custom badge detector...")
        badge_detector = YOLO(str(BADGE_MODEL_PATH))
        badge_detector.to(device)
        if use_half:
            badge_detector.model.half()
    else:
        print("⚠️  Badge detector model not found. Badge detection disabled.")
        print(f"   Train model and place at: {BADGE_MODEL_PATH}")

    # Load staff database if available
    if STAFF_DB_PATH.exists():
        print("Loading staff database...")
        with open(STAFF_DB_PATH, "rb") as f:
            staff_database = pickle.load(f)
        print(f"   Loaded {len(staff_database.get('encodings', []))} staff members")
    else:
        print("⚠️  Staff database not found. Face recognition disabled.")
        print(f"   Run build_staff_db.py to create: {STAFF_DB_PATH}")


def detect_persons(frame: np.ndarray) -> List[Dict]:
    """
    Step A: Detect all persons using YOLOv8n
    Returns list of person bounding boxes with confidence scores
    """
    results = person_detector(frame, classes=[0], verbose=False)  # class 0 = person

    persons = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            persons.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

    return persons


def detect_badge_on_chest(frame: np.ndarray, person_bbox: List[int]) -> Optional[Dict]:
    """
    Step B: Detect badge on chest region (15%-55% of person height)
    Returns badge bbox if found, else None
    """
    if badge_detector is None:
        return None

    x1, y1, x2, y2 = person_bbox
    person_height = y2 - y1

    # Define chest region (10%-70% from top of person bbox) - Expanded for better detection
    chest_y1 = y1 + int(person_height * 0.10)
    chest_y2 = y1 + int(person_height * 0.70)
    chest_region = frame[chest_y1:chest_y2, x1:x2]

    if chest_region.size == 0:
        return None

    # Run badge detector on chest region with very low confidence threshold
    results = badge_detector(chest_region, conf=0.02, verbose=False)  # Very low to catch all possible badges

    chest_h, chest_w = chest_region.shape[:2]
    valid_badges = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Calculate badge size
            badge_w = bx2 - bx1
            badge_h = by2 - by1
            badge_area = badge_w * badge_h

            # Filter out invalid detections:
            # 1. Not at image edges (at least 5px from edge)
            # 2. Reasonable size (at least 20x20px, not larger than 80% of chest)
            # 3. Aspect ratio reasonable (height > width/3, to allow for lanyard)
            if (bx1 > 5 and by1 > 5 and
                bx2 < chest_w - 5 and by2 < chest_h - 5 and
                badge_w >= 20 and badge_h >= 20 and
                badge_area < (chest_w * chest_h * 0.8) and
                badge_h > badge_w / 3):

                valid_badges.append({
                    "bbox": [x1 + bx1, chest_y1 + by1, x1 + bx2, chest_y1 + by2],
                    "confidence": conf,
                    "area": badge_area
                })

    if valid_badges:
        # Get badge with highest confidence
        best_badge = max(valid_badges, key=lambda b: b["confidence"])
        print(f"🔍 Badge detected! Confidence: {best_badge['confidence']:.2f}")  # Debug output
        return best_badge

    print(f"⚠️  No badge detected in chest region")  # Debug output
    return None


def recognize_face(frame: np.ndarray, person_bbox: List[int]) -> Optional[Dict]:
    """
    Step C: Run face recognition on head region
    Returns staff info if recognized, else None
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return None

    if staff_database is None or len(staff_database.get("encodings", [])) == 0:
        return None

    x1, y1, x2, y2 = person_bbox

    # Extract person region
    person_region = frame[y1:y2, x1:x2]

    if person_region.size == 0:
        return None

    # Convert BGR to RGB for face_recognition
    rgb_person = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)

    # Find faces in entire person region (more reliable than just head)
    face_locations = face_recognition.face_locations(rgb_person, model="hog")

    if len(face_locations) == 0:
        return None

    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_person, face_locations)

    if len(face_encodings) == 0:
        return None

    # Compare with staff database
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(
        staff_database["encodings"],
        face_encoding,
        tolerance=0.6
    )

    # Find best match
    if True in matches:
        face_distances = face_recognition.face_distance(
            staff_database["encodings"],
            face_encoding
        )
        best_match_idx = np.argmin(face_distances)

        if matches[best_match_idx]:
            return {
                "name": staff_database["names"][best_match_idx],
                "role": staff_database.get("roles", ["Unknown"] * len(matches))[best_match_idx],
                "confidence": float(1 - face_distances[best_match_idx])
            }

    return None


def process_frame(frame_data: bytes) -> Tuple[bytes, Dict]:
    """
    Main processing pipeline:
    1. Detect persons
    2. For each person: detect badge on chest + recognize face
    3. Flag violations (staff without badge)
    4. Annotate frame and return results
    """
    start_time = time.time()

    # Decode image
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return frame_data, {"error": "Failed to decode frame"}

    # Step A: Detect persons
    persons = detect_persons(frame)

    # Process each person
    staff_count = 0
    violations = []

    for idx, person in enumerate(persons):
        bbox = person["bbox"]
        x1, y1, x2, y2 = bbox

        # Step B: Detect badge
        badge = detect_badge_on_chest(frame, bbox)

        # Step C: Recognize face
        staff_info = recognize_face(frame, bbox)

        # Determine status
        is_staff = staff_info is not None
        has_badge = badge is not None

        if is_staff:
            staff_count += 1

        # Flag violation: Staff without badge
        if is_staff and not has_badge:
            violations.append({
                "person_id": idx,
                "name": staff_info["name"],
                "reason": "No badge detected"
            })

        # Annotate frame
        color = (0, 0, 255) if (is_staff and not has_badge) else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add label
        label_parts = []
        if is_staff:
            label_parts.append(f"{staff_info['name']} ({staff_info['role']})")
        else:
            label_parts.append(f"Person {idx + 1}")

        if has_badge:
            label_parts.append("✓ Badge")
        elif is_staff:
            label_parts.append("✗ NO BADGE")

        label = " | ".join(label_parts)

        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw badge bbox if detected
        if has_badge:
            bx1, by1, bx2, by2 = badge["bbox"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

    # Encode frame back to JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_frame = buffer.tobytes()

    # Calculate processing time
    process_time = (time.time() - start_time) * 1000  # ms

    # Build stats
    stats = {
        "persons_count": len(persons),
        "staff_count": staff_count,
        "violations_count": len(violations),
        "violations": violations,
        "process_time_ms": round(process_time, 2),
        "fps": round(1000 / process_time if process_time > 0 else 0, 2)
    }

    return annotated_frame, stats


async def frame_processor():
    """Background task that processes frames from the queue"""
    global processing

    while True:
        frame_data = None

        with queue_lock:
            if len(frame_queue) > 0:
                # Get newest frame and clear queue (drop old frames)
                frame_data = frame_queue.pop()
                frame_queue.clear()

        if frame_data is not None:
            processing = True
            try:
                # Extract the actual bytes from the container dict
                annotated_frame, stats = process_frame(frame_data["frame_data"])
                # Store result for WebSocket to send
                frame_data["result"] = {
                    "frame": annotated_frame,
                    "stats": stats
                }
            except Exception as e:
                print(f"Error processing frame: {e}")
                frame_data["result"] = {
                    "error": str(e)
                }
            processing = False

        await asyncio.sleep(0.01)  # Small sleep to prevent busy loop


@app.on_event("startup")
async def startup_event():
    """Load models on server startup"""
    load_models()
    # Start background frame processor
    asyncio.create_task(frame_processor())


@app.get("/")
async def root():
    """Serve the main HTML interface"""
    return FileResponse("index.html")


@app.get("/status")
async def status():
    """Health check endpoint"""
    return {
        "status": "running",
        "device": device,
        "half_precision": use_half,
        "badge_detector_loaded": badge_detector is not None,
        "staff_db_loaded": staff_database is not None
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time frame streaming
    Receives frames from client, processes them, and sends back annotated frames
    """
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()

            # Add to queue for processing
            result_container = {"frame_data": data, "result": None}

            with queue_lock:
                # If queue is full, drop oldest frame
                if len(frame_queue) >= FRAME_SKIP_THRESHOLD:
                    dropped = frame_queue.popleft()
                    print("⚠️  Dropped frame (queue full)")

                frame_queue.append(result_container)

            # Wait for processing (with timeout)
            max_wait = 0.5  # 500ms timeout
            wait_start = time.time()

            while result_container["result"] is None:
                if time.time() - wait_start > max_wait:
                    print("⚠️  Processing timeout")
                    break
                await asyncio.sleep(0.01)

            result = result_container.get("result")

            if result and "error" not in result:
                # Send back annotated frame + stats
                response = {
                    "frame": base64.b64encode(result["frame"]).decode("utf-8"),
                    "stats": result["stats"]
                }
                await websocket.send_json(response)
            elif result and "error" in result:
                await websocket.send_json({"error": result["error"]})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn

    # Get local IP address for mobile access
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        # Fallback: try to get IP by connecting to external address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = "your-local-ip"

    # Check if SSL certificates exist
    cert_file = Path("cert.pem")
    key_file = Path("key.pem")
    use_ssl = cert_file.exists() and key_file.exists()

    protocol = "https" if use_ssl else "http"

    print("\n" + "="*60)
    print("🚀 Staff ID Badge Detection System - Server Starting")
    print("="*60)
    print(f"Local access:  {protocol}://127.0.0.1:8000")
    print(f"Mobile access: {protocol}://{local_ip}:8000")
    if use_ssl:
        print("🔒 HTTPS enabled (self-signed certificate)")
    else:
        print("⚠️  HTTP mode - camera may not work on mobile")
    print("="*60 + "\n")

    if use_ssl:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem"
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
