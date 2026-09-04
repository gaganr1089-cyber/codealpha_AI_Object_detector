from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolo11n.pt")

# Create SORT tracker
tracker = Sort()

# Open webcam
cap = cv2.VideoCapture(0)


def generate_frames():

    while True:

        success, frame = cap.read()

        if not success:
            break

        # YOLO detection
        results = model(frame)

        detections = []
        detection_classes = []

        # Extract detections
        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                if confidence > 0.3:

                    detections.append([x1, y1, x2, y2])
                    detection_classes.append(class_id)

        # Convert detections
        if len(detections) > 0:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 4))

        # SORT tracking
        tracked_objects = tracker.update(detections)

        # Match tracked objects with YOLO classes
        for tracked in tracked_objects:

            tx1, ty1, tx2, ty2, track_id = tracked

            best_iou = 0
            best_class = None

            for i, detection in enumerate(detections):

                dx1, dy1, dx2, dy2 = detection

                ix1 = max(tx1, dx1)
                iy1 = max(ty1, dy1)
                ix2 = min(tx2, dx2)
                iy2 = min(ty2, dy2)

                intersection = (
                    max(0, ix2 - ix1) *
                    max(0, iy2 - iy1)
                )

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

            # Coordinates
            x1 = int(tx1)
            y1 = int(ty1)
            x2 = int(tx2)
            y2 = int(ty2)

            track_id = int(track_id)

            # Object name
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

            # Label
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

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        # Send frame to browser
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=False)