"""
Build Staff Database - Extract face encodings from staff photos
Creates a pickle file with face encodings, names, and roles for face recognition

Usage:
    python build_staff_db.py

Folder structure:
    staff_photos/
        john_doe.jpg        # Format: firstname_lastname.jpg
        jane_smith.png
        bob_wilson.jpg
        ...

Optional: Create staff_roles.json for role mapping:
    {
        "john_doe": "Security Guard",
        "jane_smith": "Manager",
        "bob_wilson": "Staff"
    }
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import cv2
import face_recognition
import numpy as np

# Configuration
STAFF_PHOTOS_DIR = Path("staff_photos")
STAFF_ROLES_FILE = Path("staff_roles.json")
OUTPUT_DB_FILE = Path("staff_db.pkl")

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_staff_roles() -> Dict[str, str]:
    """Load staff roles from JSON file if available"""
    if STAFF_ROLES_FILE.exists():
        with open(STAFF_ROLES_FILE, "r") as f:
            return json.load(f)
    return {}


def format_name(filename: str) -> str:
    """
    Convert filename to display name
    Examples:
        john_doe.jpg -> John Doe
        jane_smith.png -> Jane Smith
    """
    name = filename.stem  # Remove extension
    parts = name.split("_")
    return " ".join(part.capitalize() for part in parts)


def extract_face_encoding(image_path: Path) -> np.ndarray:
    """
    Extract face encoding from image
    Returns encoding or None if no face found
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"   ✗ Failed to load image: {image_path.name}")
        return None

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find face locations
    face_locations = face_recognition.face_locations(rgb_image, model="hog")

    if len(face_locations) == 0:
        print(f"   ✗ No face found in: {image_path.name}")
        return None

    if len(face_locations) > 1:
        print(f"   ⚠️  Multiple faces found in {image_path.name}, using first face")

    # Get face encoding
    encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if len(encodings) == 0:
        print(f"   ✗ Failed to encode face in: {image_path.name}")
        return None

    return encodings[0]


def build_database():
    """
    Build staff database from photos folder
    Saves encodings, names, and roles to pickle file
    """
    print("=" * 60)
    print("Building Staff Database")
    print("=" * 60)

    # Check if staff photos directory exists
    if not STAFF_PHOTOS_DIR.exists():
        print(f"\n✗ Error: Staff photos directory not found: {STAFF_PHOTOS_DIR}")
        print("\nPlease create the directory and add staff photos:")
        print(f"   mkdir {STAFF_PHOTOS_DIR}")
        print(f"   # Add photos in format: firstname_lastname.jpg")
        return

    # Get all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(STAFF_PHOTOS_DIR.glob(f"*{ext}"))

    if len(image_files) == 0:
        print(f"\n✗ Error: No images found in {STAFF_PHOTOS_DIR}")
        print(f"   Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        return

    print(f"\nFound {len(image_files)} image(s) in {STAFF_PHOTOS_DIR}")

    # Load roles if available
    roles_map = load_staff_roles()
    if STAFF_ROLES_FILE.exists():
        print(f"Loaded roles from {STAFF_ROLES_FILE}")
    else:
        print(f"No roles file found, using default role 'Staff'")

    # Process each image
    encodings = []
    names = []
    roles = []

    print("\nProcessing images...")
    for image_path in sorted(image_files):
        print(f"\n{image_path.name}:")

        # Extract face encoding
        encoding = extract_face_encoding(image_path)

        if encoding is not None:
            # Get name from filename
            name = format_name(image_path)
            # Get role from roles map or use default
            role_key = image_path.stem  # filename without extension
            role = roles_map.get(role_key, "Staff")

            encodings.append(encoding)
            names.append(name)
            roles.append(role)

            print(f"   ✓ Added: {name} ({role})")

    # Save database
    if len(encodings) == 0:
        print("\n✗ Error: No faces were successfully encoded")
        return

    database = {
        "encodings": encodings,
        "names": names,
        "roles": roles
    }

    with open(OUTPUT_DB_FILE, "wb") as f:
        pickle.dump(database, f)

    print("\n" + "=" * 60)
    print(f"✓ Database saved to: {OUTPUT_DB_FILE}")
    print(f"  Total staff members: {len(encodings)}")
    print("=" * 60)

    # Show summary
    print("\nStaff Summary:")
    for name, role in zip(names, roles):
        print(f"  • {name}: {role}")

    print("\nTo use this database:")
    print("  1. Make sure staff_db.pkl is in the same directory as server.py")
    print("  2. Start the server: python server.py")


def create_example_roles_file():
    """Create an example staff_roles.json file"""
    if STAFF_ROLES_FILE.exists():
        print(f"Roles file already exists: {STAFF_ROLES_FILE}")
        return

    example_roles = {
        "john_doe": "Security Guard",
        "jane_smith": "Manager",
        "bob_wilson": "Staff"
    }

    with open(STAFF_ROLES_FILE, "w") as f:
        json.dump(example_roles, f, indent=4)

    print(f"Created example roles file: {STAFF_ROLES_FILE}")
    print("Edit this file to assign roles to your staff members")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build staff database from photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  1. Create staff photos directory:
     mkdir staff_photos

  2. Add staff photos (format: firstname_lastname.jpg):
     cp john_doe.jpg staff_photos/
     cp jane_smith.png staff_photos/

  3. (Optional) Create staff_roles.json:
     python build_staff_db.py --create-roles
     # Then edit staff_roles.json to assign roles

  4. Build database:
     python build_staff_db.py

  5. Database will be saved as staff_db.pkl
        """
    )

    parser.add_argument(
        "--create-roles",
        action="store_true",
        help="Create an example staff_roles.json file"
    )

    args = parser.parse_args()

    if args.create_roles:
        create_example_roles_file()
    else:
        build_database()
