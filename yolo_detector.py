from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import cv2

#
text = "Phanlop-Clicknext-Internship-2024"

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize track history
track_history = []

def draw_track_line(frame, boxes, track_history) -> None:
    for box in boxes.xywh.cpu():
        x, y, w, h = box
        track_history.append((float(x), float(y)))
        if len(track_history) > 30:
            track_history.pop(0)
        
        points = np.hstack(track_history).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf

    # Draw bounding box
    annotator.box_label(
        box=coordinator, label=class_name, color=colors(class_id, True)
    )

    return annotator.result()


def detect_object(frame, track_history):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model.track(frame, persist=True, classes=[15], )

    for result in results:
        #print("result[1]: ", result[1])
        if result: 
            frame = draw_boxes(frame, result.boxes)
            draw_track_line(frame, result.boxes, track_history)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    video_writer = cv2.VideoWriter(
        video_path + "_demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)
    )

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if ret:
            # Detect motorcycle from image frame
            frame_result = detect_object(frame, track_history)

            # Write result to video
            video_writer.write(frame_result)

            # Writ text to video
            
            cv2.putText(frame, 
                        text,
                        (600, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_4)

            # Show result
            cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
            cv2.imshow("Video", frame_result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break

    # Release the VideoCapture object and close the window
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
