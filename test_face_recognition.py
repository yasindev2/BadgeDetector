"""Test face recognition with staff database"""
import pickle
import cv2
import face_recognition

# Load database
with open("staff_db.pkl", "rb") as f:
    db = pickle.load(f)

print("Database loaded:")
print(f"  Staff: {db['names']}")
print(f"  Encodings: {len(db['encodings'])}")

# Test with one of the original photos
test_image_path = "staff_photos/muhammad_yasin.jpg"
image = cv2.imread(test_image_path)

if image is None:
    print(f"Failed to load {test_image_path}")
    exit(1)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Find faces
face_locations = face_recognition.face_locations(rgb_image, model="hog")
print(f"\nFace locations found: {len(face_locations)}")

if len(face_locations) == 0:
    print("No faces found in test image!")
    exit(1)

# Get encodings
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
print(f"Face encodings: {len(face_encodings)}")

if len(face_encodings) == 0:
    print("No face encodings generated!")
    exit(1)

# Compare with database
test_encoding = face_encodings[0]
matches = face_recognition.compare_faces(db["encodings"], test_encoding, tolerance=0.6)
print(f"\nMatches: {matches}")

if True in matches:
    distances = face_recognition.face_distance(db["encodings"], test_encoding)
    print(f"Face distances: {distances}")
    best_match_idx = distances.argmin()
    print(f"\n✓ Recognized as: {db['names'][best_match_idx]}")
    print(f"  Confidence: {1 - distances[best_match_idx]:.2%}")
else:
    print("\n✗ No match found")
