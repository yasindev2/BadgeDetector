"""
Train Custom Badge Detection Model using YOLOv8n

This script trains a YOLOv8 model to detect ID badges/lanyards on chest region.

Usage:
    python train_badge_model.py

Dataset Structure (YOLO format):
    badge_dataset/
        data.yaml           # Dataset configuration
        images/
            train/          # Training images
                img001.jpg
                img002.jpg
                ...
            val/            # Validation images
                img101.jpg
                img102.jpg
                ...
        labels/
            train/          # Training labels (YOLO format)
                img001.txt
                img002.txt
                ...
            val/            # Validation labels
                img101.txt
                img102.txt
                ...

YOLO Label Format (one line per object):
    <class_id> <x_center> <y_center> <width> <height>

    Where:
        class_id: 0 (for badge)
        x_center, y_center, width, height: normalized (0-1) relative to image size

    Example: 0 0.5 0.3 0.1 0.15
    (Badge at center-x 50%, center-y 30%, width 10%, height 15%)

data.yaml Format:
    path: /path/to/badge_dataset
    train: images/train
    val: images/val
    nc: 1
    names: ['badge']
"""

from pathlib import Path

from ultralytics import YOLO

# Configuration
DATASET_DIR = Path("badge_dataset")
DATA_YAML = DATASET_DIR / "data.yaml"
OUTPUT_DIR = Path("models")
MODEL_NAME = "badge_detector.pt"

# Training parameters
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
LEARNING_RATE = 0.01
PATIENCE = 20  # Early stopping patience


def create_example_dataset_structure():
    """Create example dataset directory structure and data.yaml"""
    print("Creating example dataset structure...")

    # Create directories
    dirs = [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "labels" / "train",
        DATASET_DIR / "labels" / "val"
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")

    # Create data.yaml
    yaml_content = f"""# Badge Detection Dataset Configuration

path: {DATASET_DIR.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['badge']  # class names
"""

    with open(DATA_YAML, "w") as f:
        f.write(yaml_content)

    print(f"\n  Created: {DATA_YAML}")

    # Create README in dataset folder
    readme_content = """# Badge Detection Dataset

## Collecting Images

1. Collect images of people wearing ID badges/lanyards
2. Focus on chest region (where badges typically hang)
3. Include variety:
   - Different badge types (vertical, horizontal)
   - Different lighting conditions
   - Different distances/angles
   - With and without badges (for negative examples)

Recommended: 200+ training images, 50+ validation images

## Annotating Images

Use a YOLO annotation tool like:
- Roboflow: https://roboflow.com (recommended, has free tier)
- LabelImg: https://github.com/heartexlabs/labelImg
- CVAT: https://github.com/opencv/cvat

Annotation tips:
- Draw bounding box around the badge/lanyard
- Include the full badge and lanyard in the box
- Be consistent with annotation style
- Label class: "badge"

## Label Format

Each image needs a corresponding .txt file with same name in labels/ folder.

Format: <class_id> <x_center> <y_center> <width> <height>

Example (badge_001.txt):
0 0.512 0.325 0.089 0.142

All values normalized to 0-1 range.

## Quick Start with Roboflow

1. Sign up at https://roboflow.com
2. Create new project (Object Detection)
3. Upload your images
4. Annotate badges as "badge" class
5. Export in "YOLOv8" format
6. Extract to badge_dataset/ folder
7. Run: python train_badge_model.py
"""

    with open(DATASET_DIR / "README.md", "w") as f:
        f.write(readme_content)

    print(f"  Created: {DATASET_DIR / 'README.md'}")

    print("\n✓ Dataset structure created!")
    print("\nNext steps:")
    print("  1. Add training images to: badge_dataset/images/train/")
    print("  2. Add validation images to: badge_dataset/images/val/")
    print("  3. Add corresponding labels to: badge_dataset/labels/train/ and val/")
    print("  4. Run: python train_badge_model.py")


def check_dataset():
    """Verify dataset structure and count samples"""
    print("Checking dataset...")

    if not DATASET_DIR.exists():
        print(f"✗ Dataset directory not found: {DATASET_DIR}")
        return False

    if not DATA_YAML.exists():
        print(f"✗ data.yaml not found: {DATA_YAML}")
        return False

    # Count training images
    train_images = list((DATASET_DIR / "images" / "train").glob("*.[jp][pn]g"))
    train_labels = list((DATASET_DIR / "labels" / "train").glob("*.txt"))

    # Count validation images
    val_images = list((DATASET_DIR / "images" / "val").glob("*.[jp][pn]g"))
    val_labels = list((DATASET_DIR / "labels" / "val").glob("*.txt"))

    print(f"\nDataset Statistics:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Training labels: {len(train_labels)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Validation labels: {len(val_labels)}")

    # Warnings
    if len(train_images) == 0:
        print("\n⚠️  No training images found!")
        return False

    if len(val_images) == 0:
        print("\n⚠️  No validation images found!")
        return False

    if len(train_images) < 50:
        print("\n⚠️  Warning: Less than 50 training images. Recommended: 200+")

    if len(train_labels) < len(train_images):
        print(f"\n⚠️  Warning: Some training images missing labels")
        print(f"     {len(train_images) - len(train_labels)} images without labels")

    return True


def train_model():
    """Train YOLOv8 badge detection model"""
    print("\n" + "=" * 60)
    print("Training Badge Detection Model")
    print("=" * 60)

    # Check dataset
    if not check_dataset():
        print("\n✗ Dataset check failed. Please fix issues and try again.")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load pre-trained YOLOv8n model
    print("\nLoading YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # Start from pre-trained COCO model

    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {PATIENCE}")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        device=0 if __import__("torch").cuda.is_available() else "cpu",
        project=str(OUTPUT_DIR),
        name="badge_training",
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True
    )

    # Get best model path
    best_model_path = OUTPUT_DIR / "badge_training" / "weights" / "best.pt"

    if best_model_path.exists():
        # Copy to final location
        import shutil
        final_model_path = OUTPUT_DIR / MODEL_NAME
        shutil.copy(best_model_path, final_model_path)

        print("\n" + "=" * 60)
        print("✓ Training completed!")
        print("=" * 60)
        print(f"\nBest model saved to: {final_model_path}")
        print(f"Training results: {OUTPUT_DIR / 'badge_training'}")
        print("\nNext steps:")
        print("  1. Check training plots in models/badge_training/")
        print("  2. Test model: python test_badge_model.py")
        print("  3. Use in server: python server.py")
    else:
        print("\n✗ Training failed - best model not found")


def test_trained_model():
    """Quick test of trained model"""
    model_path = OUTPUT_DIR / MODEL_NAME

    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("  Train model first: python train_badge_model.py")
        return

    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))

    # Test on validation images
    val_images = list((DATASET_DIR / "images" / "val").glob("*.[jp][pn]g"))

    if len(val_images) == 0:
        print("✗ No validation images found")
        return

    print(f"\nTesting on {len(val_images)} validation images...")
    results = model.val(data=str(DATA_YAML))

    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.3f}")
    print(f"  mAP50-95: {results.box.map:.3f}")
    print(f"  Precision: {results.box.mp:.3f}")
    print(f"  Recall: {results.box.mr:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train YOLOv8 badge detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create example dataset structure"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test trained model on validation set"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )

    args = parser.parse_args()

    # Update globals if arguments provided
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch:
        BATCH_SIZE = args.batch

    if args.setup:
        create_example_dataset_structure()
    elif args.test:
        test_trained_model()
    else:
        train_model()
