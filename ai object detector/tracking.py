from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort

print("Setup successful")

# Load YOLO model
model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully")

# Create SORT tracker
# max_age: how many frames an object can disappear before its ID is removed
# min_hits: detections needed before a new track is confirmed
# iou_threshold: how much bounding boxes need to overlap
tracker = Sort(
    max_age=30,
    min_hits=3,
    iou_threshold=0.2
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    detections = []
    detection_classes = []

    # Extract YOLO detections
    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if confidence > 0.4:

                detections.append([
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence
                ])

                detection_classes.append(class_id)

    # Convert detections to NumPy array
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    # Run SORT tracking
    tracked_objects = tracker.update(detections)

    # Match tracked objects with YOLO detections
    for tracked in tracked_objects:

        tx1, ty1, tx2, ty2, track_id = tracked

        best_iou = 0
        best_class = None

        for i, detection in enumerate(detections):

            dx1, dy1, dx2, dy2 = detection[:4]

            # Calculate intersection
            ix1 = max(tx1, dx1)
            iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2)
            iy2 = min(ty2, dy2)

            intersection = (
                max(0, ix2 - ix1) *
                max(0, iy2 - iy1)
            )

            # Calculate areas
            tracked_area = (
                max(0, tx2 - tx1) *
                max(0, ty2 - ty1)
            )

            detection_area = (
                max(0, dx2 - dx1) *
                max(0, dy2 - dy1)
            )

            union = (
                tracked_area +
                detection_area -
                intersection
            )

            if union > 0:

                current_iou = intersection / union

                if current_iou > best_iou:

                    best_iou = current_iou
                    best_class = detection_classes[i]

        # Convert coordinates
        x1 = int(tx1)
        y1 = int(ty1)
        x2 = int(tx2)
        y2 = int(ty2)
        track_id = int(track_id)

        # Get class name
        if best_class is not None:
            class_name = model.names[best_class]
        else:
            class_name = "Object"

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display label
        label = f"{class_name} | ID: {track_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display frame
    cv2.imshow(
        "YOLO + SORT Object Tracking",
        frame
    )

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Program ended successfully")