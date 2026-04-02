from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2

#
text = "Phanlop-Clicknext-Internship-2024"

# Load YOLO model
model = YOLO("yolov8n.pt")

def draw_track_line(frame, boxes, track_line_vec):
    """"Draw track line on each from track_line_vec"""
    

def bounding_boxes_center(coordinator, track_line_vec):
    """Calculate bounding box center"""
    x_center = coordinator[0] + (coordinator[2]/2)
    y_center = coordinator[1] + (coordinator[3]/2)

    xy_center = [x_center, y_center]
    track_line_vec.append(xy_center)
    return None

def draw_boxes(frame, boxes, track_line_vec):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        track_line_vec.append()
        confidence = box.conf

    # Draw bounding box
    annotator.box_label(
        box=coordinator, label=class_name, color=colors(class_id, True)
    )

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model.predict(frame, classes=[15])

    for result in results:
        #print("result[1]: ", result[1])
        if result: 
            frame = draw_boxes(frame, result.boxes)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    video_writer = cv2.VideoWriter(
        video_path + "_demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 60, (1280, 720)
    )

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()
        track_line_vec = []

        if ret:
            # Detect motorcycle from image frame
            frame_result = detect_object(frame, track_line_vec)



            # Write result to video
            video_writer.write(frame_result)

            # Writ text to video
            # fps = cap.get(cv2.CAP_PROP_FPS)

            cv2.line()
            
            cv2.putText(frame, 
                        "fps: " + str(fps) + text,
                        (550, 40),
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
