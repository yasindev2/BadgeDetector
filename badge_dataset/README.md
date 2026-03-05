# Badge Detection Dataset

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
