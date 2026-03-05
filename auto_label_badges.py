"""
Auto-generate badge labels for training images
Creates approximate YOLO labels based on chest region detection
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Load person detector
print("Loading YOLOv8 person detector...")
person_detector = YOLO("yolov8n.pt")

# Paths
train_images_dir = Path("badge_dataset/images/train")
train_labels_dir = Path("badge_dataset/labels/train")
train_labels_dir.mkdir(parents=True, exist_ok=True)

# Get all images without labels
all_images = list(train_images_dir.glob("*.jpg"))
existing_labels = {f.stem for f in train_labels_dir.glob("*.txt")}
unlabeled_images = [img for img in all_images if img.stem not in existing_labels]

print(f"\nFound {len(unlabeled_images)} unlabeled images")
print(f"Generating approximate badge labels...")

labeled_count = 0

for img_path in unlabeled_images:
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ✗ Failed to read: {img_path.name}")
        continue

    h, w = img.shape[:2]

    # Detect persons
    results = person_detector(img, classes=[0], verbose=False)

    # Find first person
    person_found = False
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            # Get first person's bbox
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_height = y2 - y1

            # Define chest region (15%-55% from top of person)
            chest_y1 = y1 + int(person_height * 0.15)
            chest_y2 = y1 + int(person_height * 0.55)
            chest_height = chest_y2 - chest_y1

            # Approximate badge position (center of chest, smaller width)
            badge_width = int((x2 - x1) * 0.25)  # 25% of person width
            badge_height = int(chest_height * 0.4)  # 40% of chest height
            badge_x_center = (x1 + x2) / 2
            badge_y_center = (chest_y1 + chest_y2) / 2

            # Convert to YOLO format (normalized 0-1)
            x_center_norm = badge_x_center / w
            y_center_norm = badge_y_center / h
            width_norm = badge_width / w
            height_norm = badge_height / h

            # Clamp values to 0-1 range
            x_center_norm = max(0, min(1, x_center_norm))
            y_center_norm = max(0, min(1, y_center_norm))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))

            # Write label file
            label_path = train_labels_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

            print(f"  ✓ {img_path.name}")
            labeled_count += 1
            person_found = True
            break

    if not person_found:
        print(f"  ⚠️  No person detected in: {img_path.name}")

print(f"\n{'='*60}")
print(f"✓ Auto-labeled {labeled_count} images")
print(f"Total labels: {len(list(train_labels_dir.glob('*.txt')))}")
print(f"{'='*60}")
print("\nNote: These are approximate labels. For best results:")
print("  1. Review and adjust labels using an annotation tool")
print("  2. Or train with these and refine later")
print("\nReady to train!")
