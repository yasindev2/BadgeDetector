"""
System Test Script - Verify all components are working

This script tests each component of the badge detection system.
"""

import sys
from pathlib import Path


def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    errors = []

    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("cv2", "OpenCV"),
        ("ultralytics", "YOLOv8"),
        ("face_recognition", "face_recognition"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("torch", "PyTorch"),
    ]

    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name} - {e}")
            errors.append(display_name)

    if errors:
        print(f"\n✗ Missing packages: {', '.join(errors)}")
        print("  Run: pip install -r requirements.txt")
        return False

    print("✓ All packages installed\n")
    return True


def test_yolo():
    """Test if YOLOv8 can be loaded"""
    print("Testing YOLOv8...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("  ✓ YOLOv8n loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load YOLOv8: {e}")
        return False


def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ⚠️  No GPU available (will use CPU)")
            print("     For better performance, install CUDA-enabled PyTorch")
            return False
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        return False


def test_staff_db():
    """Test if staff database exists"""
    print("\nTesting staff database...")
    db_path = Path("staff_db.pkl")

    if not db_path.exists():
        print("  ⚠️  Staff database not found")
        print("     Face recognition will be disabled")
        print("     Create database: python build_staff_db.py")
        return False

    try:
        import pickle
        with open(db_path, "rb") as f:
            db = pickle.load(f)

        count = len(db.get("encodings", []))
        print(f"  ✓ Staff database loaded: {count} staff members")

        # Show staff list
        names = db.get("names", [])
        roles = db.get("roles", [])
        if names:
            print("\n  Staff members:")
            for name, role in zip(names, roles):
                print(f"    • {name}: {role}")

        return True
    except Exception as e:
        print(f"  ✗ Error loading staff database: {e}")
        return False


def test_badge_model():
    """Test if badge detection model exists"""
    print("\nTesting badge detection model...")
    model_path = Path("models/badge_detector.pt")

    if not model_path.exists():
        print("  ⚠️  Badge detection model not found")
        print("     Badge detection will be disabled")
        print("     Train model: python train_badge_model.py")
        return False

    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print("  ✓ Badge detection model loaded")
        return True
    except Exception as e:
        print(f"  ✗ Error loading badge model: {e}")
        return False


def test_webcam():
    """Test if webcam is accessible"""
    print("\nTesting webcam access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("  ✗ Cannot open webcam")
            print("     Check if camera is being used by another application")
            return False

        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  ✓ Webcam accessible: {w}x{h}")
            return True
        else:
            print("  ✗ Cannot read frame from webcam")
            return False

    except Exception as e:
        print(f"  ✗ Error accessing webcam: {e}")
        return False


def test_network():
    """Show network information for mobile access"""
    print("\nNetwork information...")
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        print(f"  Hostname: {hostname}")
        print(f"  Local IP: {local_ip}")
        print(f"\n  Access URLs:")
        print(f"    Local:  http://127.0.0.1:8000")
        print(f"    Mobile: http://{local_ip}:8000")

        return True
    except Exception as e:
        print(f"  ✗ Error getting network info: {e}")
        return False


def run_all_tests():
    """Run all system tests"""
    print("=" * 60)
    print("Staff Badge Detection System - System Test")
    print("=" * 60)
    print()

    results = {
        "Imports": test_imports(),
        "YOLOv8": test_yolo(),
        "GPU": test_gpu(),
        "Staff Database": test_staff_db(),
        "Badge Model": test_badge_model(),
        "Webcam": test_webcam(),
        "Network": test_network(),
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    critical_tests = ["Imports", "YOLOv8"]
    optional_tests = ["GPU", "Staff Database", "Badge Model"]

    for test_name, result in results.items():
        if test_name in critical_tests:
            status = "✓ PASS" if result else "✗ FAIL (CRITICAL)"
        elif test_name in optional_tests:
            status = "✓ READY" if result else "⚠️  NOT CONFIGURED (optional)"
        else:
            status = "✓ OK" if result else "⚠️  WARNING"

        print(f"  {test_name:20s}: {status}")

    # Determine if system is ready
    critical_passed = all(results[t] for t in critical_tests)

    print("\n" + "=" * 60)

    if critical_passed:
        print("✓ System is ready to run!")
        print("\nStart server with:")
        print("  python server.py")
        print("  # OR")
        print("  ./start_server.sh")

        if not results["Staff Database"]:
            print("\nOptional: Enable face recognition")
            print("  1. mkdir staff_photos")
            print("  2. Add photos: firstname_lastname.jpg")
            print("  3. python build_staff_db.py")

        if not results["Badge Model"]:
            print("\nOptional: Enable badge detection")
            print("  1. python train_badge_model.py --setup")
            print("  2. Annotate images in badge_dataset/")
            print("  3. python train_badge_model.py")

    else:
        print("✗ System is not ready")
        print("\nPlease fix critical errors above")
        print("  Run: pip install -r requirements.txt")

    print("=" * 60)

    return critical_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
