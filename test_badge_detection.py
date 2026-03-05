"""
Test badge detection model on training images
Shows what the model is actually detecting
"""
import cv2
from pathlib import Path
from ultralytics import YOLO

# Load models
print("Loading models...")
person_detector = YOLO("yolov8n.pt")
badge_detector = YOLO("models/badge_detector.pt")

# Test on a training image - get the first available training image
train_images = list(Path("badge_dataset/images/train").glob("*.jpg"))
if len(train_images) == 0:
    print("❌ No training images found")
    exit(1)

test_image_path = train_images[0]

print(f"\nTesting on: {test_image_path}")

# Read image
img = cv2.imread(str(test_image_path))
h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

# Detect person
print("\n1. Detecting person...")
person_results = person_detector(img, classes=[0], verbose=False)

for r in person_results:
    boxes = r.boxes
    if len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_conf = float(box.conf[0])
        person_height = y2 - y1

        print(f"   ✓ Person detected at ({x1},{y1}) to ({x2},{y2})")
        print(f"   Person height: {person_height}px, confidence: {person_conf:.2f}")

        # Draw person box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Define chest region
        chest_y1 = y1 + int(person_height * 0.15)
        chest_y2 = y1 + int(person_height * 0.55)
        chest_region = img[chest_y1:chest_y2, x1:x2]

        print(f"\n2. Checking chest region: ({x1},{chest_y1}) to ({x2},{chest_y2})")
        print(f"   Chest region size: {chest_region.shape[1]}x{chest_region.shape[0]}")

        # Draw chest region
        cv2.rectangle(img, (x1, chest_y1), (x2, chest_y2), (255, 0, 255), 2)
        cv2.putText(img, "Chest Region", (x1, chest_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Detect badge in chest region with very low confidence
        print("\n3. Running badge detection on chest region...")
        badge_results = badge_detector(chest_region, conf=0.01, verbose=True)  # Very low threshold

        badges_found = 0
        for br in badge_results:
            badge_boxes = br.boxes
            print(f"   Number of detections: {len(badge_boxes)}")

            for bbox in badge_boxes:
                bx1, by1, bx2, by2 = map(int, bbox.xyxy[0])
                badge_conf = float(bbox.conf[0])

                print(f"   ✓ Badge detected! Confidence: {badge_conf:.2f}")
                print(f"     Position in chest: ({bx1},{by1}) to ({bx2},{by2})")

                # Convert to full frame coordinates
                full_bx1 = x1 + bx1
                full_by1 = chest_y1 + by1
                full_bx2 = x1 + bx2
                full_by2 = chest_y1 + by2

                print(f"     Position in frame: ({full_bx1},{full_by1}) to ({full_bx2},{full_by2})")

                # Draw badge box
                cv2.rectangle(img, (full_bx1, full_by1), (full_bx2, full_by2), (0, 0, 255), 2)
                cv2.putText(img, f"Badge {badge_conf:.2f}", (full_bx1, full_by1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                badges_found += 1

        if badges_found == 0:
            print("   ❌ No badges detected!")
            print("\n   Possible reasons:")
            print("   - Auto-generated labels were inaccurate")
            print("   - Badge is outside the chest region (15%-55%)")
            print("   - Model needs retraining with better labels")

# Save result
output_path = "badge_detection_test.jpg"
cv2.imwrite(output_path, img)
print(f"\n✓ Result saved to: {output_path}")
print("  Open this image to see what was detected")
print(f"\nTo view: open {output_path}")
