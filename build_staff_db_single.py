"""
Build staff database for a single person with multiple photos
"""
import pickle
from pathlib import Path
import cv2
import face_recognition
import numpy as np

STAFF_PHOTOS_DIR = Path("staff_photos")
OUTPUT_DB_FILE = Path("staff_db.pkl")
PERSON_NAME = "Muhammad Yasin"
PERSON_ROLE = "Staff"

print("=" * 60)
print(f"Building Staff Database for: {PERSON_NAME}")
print("=" * 60)

# Get all image files
image_files = list(STAFF_PHOTOS_DIR.glob("*.jpg"))
image_files.extend(STAFF_PHOTOS_DIR.glob("*.png"))

print(f"\nFound {len(image_files)} image(s)")

encodings = []

print("\nProcessing images...")
for image_path in sorted(image_files):
    print(f"\n{image_path.name}:")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"   ✗ Failed to load image")
        continue

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find face locations
    face_locations = face_recognition.face_locations(rgb_image, model="hog")

    if len(face_locations) == 0:
        print(f"   ✗ No face found")
        continue

    # Get face encoding
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if len(face_encodings) == 0:
        print(f"   ✗ Failed to encode face")
        continue

    encodings.append(face_encodings[0])
    print(f"   ✓ Face encoding extracted")

if len(encodings) == 0:
    print("\n✗ Error: No faces were successfully encoded")
    exit(1)

# Average the encodings for better recognition
average_encoding = np.mean(encodings, axis=0)

# Save database with single person
database = {
    "encodings": [average_encoding],
    "names": [PERSON_NAME],
    "roles": [PERSON_ROLE]
}

with open(OUTPUT_DB_FILE, "wb") as f:
    pickle.dump(database, f)

print("\n" + "=" * 60)
print(f"✓ Database saved to: {OUTPUT_DB_FILE}")
print(f"  Staff member: {PERSON_NAME}")
print(f"  Role: {PERSON_ROLE}")
print(f"  Face encodings averaged from {len(encodings)} photos")
print("=" * 60)
