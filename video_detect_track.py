"""
video_detect_track.py
---------------------
Fruit detection + multi-object tracking in a video using YOLOv8 + ByteTrack.

Detects : apple, banana, orange
Tracker : ByteTrack  (bytetrack.yaml – bundled with Ultralytics)
Input   : input.mp4
Output  : output.mp4  (annotated video saved to disk)
          Real-time preview window
          Total unique fruit count printed to console
"""

import time
import cv2
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH     = "yolov8n.pt"
INPUT_VIDEO    = "input.mp4"
OUTPUT_VIDEO   = "output.mp4"
TRACKER_CONFIG = "bytetrack.yaml"  # bundled with Ultralytics ≥ 8.0
CONF_THRESHOLD = 0.35

TARGET_FRUITS = {"apple", "banana", "orange"}

COLOURS = {
    "apple":  (0,  255,  0),
    "banana": (0,  215, 255),
    "orange": (0, 128,  255),
}
DEFAULT_COLOUR = (255, 0, 255)
# ─────────────────────────────────────────────────────────────────────────────


def draw_box(frame, x1, y1, x2, y2, label, colour):
    """Draw a filled-header bounding box with a class + ID label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(frame, label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def overlay_stats(frame, unique_count: int, fps: float):
    """Render unique-fruit count and FPS in the top-left corner."""
    h = frame.shape[0]

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Unique Fruits: {unique_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def process_video(video_path: str) -> int:
    """
    Run YOLOv8 tracking on every frame of *video_path*.

    Returns
    -------
    int  – total number of unique fruits tracked throughout the video.
    """
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # Video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # VideoWriter – use mp4v codec (works on all platforms)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_in, (width, height))

    # Track unique fruit IDs seen across ALL frames
    seen_ids: set[int] = set()

    frame_idx = 0
    fps_display = 0.0
    t_prev = time.time()

    print(f"Processing {total_frames} frames …  (press 'q' to abort)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── YOLOv8 Tracking ──────────────────────────────────────────────────
        # persist=True keeps tracker state between calls (essential for ByteTrack)
        results = model.track(
            frame,
            conf=CONF_THRESHOLD,
            tracker=TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )[0]

        # ── Process detections ───────────────────────────────────────────────
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id   = int(box.cls[0])
                cls_name = model.names[cls_id].lower()

                if cls_name not in TARGET_FRUITS:
                    continue

                # Tracking ID (None when tracker hasn't assigned one yet)
                track_id = int(box.id[0]) if box.id is not None else -1
                conf     = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Register this fruit's ID
                if track_id != -1:
                    seen_ids.add(track_id)

                colour = COLOURS.get(cls_name, DEFAULT_COLOUR)
                id_str = f"#{track_id}" if track_id != -1 else ""
                label  = f"{cls_name}{id_str} {conf:.2f}"
                draw_box(frame, x1, y1, x2, y2, label, colour)

        # ── FPS calculation ──────────────────────────────────────────────────
        t_now      = time.time()
        fps_display = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev      = t_now

        # ── Stats overlay ────────────────────────────────────────────────────
        overlay_stats(frame, len(seen_ids), fps_display)

        writer.write(frame)
        cv2.imshow("Fruit Tracking", frame)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Aborted by user.")
            break

        # Progress log every 30 frames
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total_frames} | "
                  f"Unique fruits so far: {len(seen_ids)} | FPS: {fps_display:.1f}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    return len(seen_ids)


def main():
    print("=" * 55)
    print("  YOLOv8 Fruit Video Detector + Tracker (ByteTrack)")
    print("=" * 55)

    unique_count = process_video(INPUT_VIDEO)

    print(f"\nOutput saved → {OUTPUT_VIDEO}")
    print(f"Total unique fruits tracked: {unique_count}")


if __name__ == "__main__":
    main()