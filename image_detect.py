"""
image_detect.py
---------------
Fruit detection in a single image using YOLOv8.

Detects: apple, banana, orange
Input  : input.jpg
Output : output_image.jpg  (annotated image saved to disk)
         Fruit counts printed to console
"""

import cv2
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH  = "yolov8n.pt"   # YOLOv8 nano – downloaded automatically on first run
INPUT_IMAGE = "Apples.jpeg"
OUTPUT_IMAGE = "output_image.jpeg"
CONF_THRESHOLD = 0.35        # minimum confidence to accept a detection

# Fruits we care about (must match COCO class names used by YOLOv8)
TARGET_FRUITS = {"apple", "banana", "orange"}

# BGR colours per class for bounding-box drawing
COLOURS = {
    "apple":  (0,  255,  0),   # green
    "banana": (0,  215, 255),  # gold
    "orange": (0, 128,  255),  # orange
}
DEFAULT_COLOUR = (255, 0, 255)  # magenta – fallback
# ─────────────────────────────────────────────────────────────────────────────


def draw_box(frame, x1, y1, x2, y2, label, colour):
    """Draw a filled-header bounding box with a class label."""
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    # Label background
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), colour, -1)

    # Label text (black for readability on bright backgrounds)
    cv2.putText(frame, label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def detect_fruits(image_path: str) -> dict:
    """
    Run YOLOv8 detection on *image_path*.

    Returns
    -------
    dict  e.g. {"apple": 2, "banana": 1, "orange": 3}
    """
    # Load model
    model = YOLO(MODEL_PATH)

    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    # Run inference (single image, no tracking needed)
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    counts: dict[str, int] = {}

    if results.boxes is None or len(results.boxes) == 0:
        print("No detections found.")
        return counts

    for box in results.boxes:
        cls_id    = int(box.cls[0])
        cls_name  = model.names[cls_id].lower()
        conf      = float(box.conf[0])

        # Skip anything that is not a target fruit
        if cls_name not in TARGET_FRUITS:
            continue

        # Bounding-box coordinates (integer pixel values)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Update count
        counts[cls_name] = counts.get(cls_name, 0) + 1

        # Draw annotation
        colour = COLOURS.get(cls_name, DEFAULT_COLOUR)
        label  = f"{cls_name} {conf:.2f}"
        draw_box(frame, x1, y1, x2, y2, label, colour)

    # ── Overlay summary text ─────────────────────────────────────────────────
    summary = "  |  ".join(f"{k}: {v}" for k, v in counts.items()) or "No fruits"
    cv2.putText(frame, f"Detected: {summary}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save & display
    cv2.imwrite(OUTPUT_IMAGE, frame)
    print(f"Output saved → {OUTPUT_IMAGE}")

    cv2.imshow("Fruit Detection", frame)
    print("Press any key in the image window to close …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return counts


def main():
    print("=" * 50)
    print("  YOLOv8 Fruit Image Detector")
    print("=" * 50)
    counts = detect_fruits(INPUT_IMAGE)
    print("\nFruit counts:", counts)
    total = sum(counts.values())
    print(f"Total fruits detected: {total}")


if __name__ == "__main__":
    main()