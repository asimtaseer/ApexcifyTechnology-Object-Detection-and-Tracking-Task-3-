
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# Load YOLOv8 Model

# Load a pre-trained YOLOv8
model = YOLO("yolov8x.pt")


# Load DeepSORT Tracker

tracker = DeepSort(max_age=30,n_init=3,embedder="mobilenet")


# Load Video


# Open input video file
cap = cv2.VideoCapture("2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc,fps,(1280,720))

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Loop over video frames
while True:

    # Read one frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing and consistent size
    frame = cv2.resize(frame, (720, 500))

    # Run YOLO model on the frame
    
    results = model(frame)[0]
    # results[0] contains detections for this frame


    # List to store detections in DeepSORT format
    detections = []

    # Loop through each detected bounding box
    for box in results.boxes:

        # Get detected class ID (0 = person in COCO dataset)
        cls = int(box.cls[0])

        # Skip all classes except person
        if cls != 0:
            continue

        # Get bounding box coordinates (top-left and bottom-right)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Calculate width of bounding box
        w = int(x2 - x1)

        # Calculate height of bounding box
        h = int(y2 - y1)

        # Get confidence score of detection
        conf = float(box.conf[0])

        # Skip weak detections
        if conf < 0.3:
            continue

        # Append detection in DeepSORT format:
        # [[x, y, w, h], confidence, class_id]
        detections.append([[int(x1), int(y1), w, h], conf, cls])


    
    # DeepSORT Tracking

    tracks = tracker.update_tracks(detections, frame=frame)


    # Draw Tracking Results

    # Loop through all active tracks
    for track in tracks:

        # Skip unconfirmed tracks
        if not track.is_confirmed():
            continue

        # Get bounding box in left-top-right-bottom format
        x1, y1, x2, y2 = track.to_ltrb()

        # Get unique ID assigned to this object
        track_id = track.track_id

        # Draw bounding box around tracked person
        cv2.rectangle(
            frame,(int(x1 - 3), int(y1 - 20)),(int(x2 + 3), int(y1)),(255, 0, 0),-1)
        cv2.rectangle(
            frame,(int(x1), int(y1)),(int(x2), int(y2)),(255, 0, 0),3)

        # Draw ID label above the bounding box
        cv2.putText(
            frame,f"ID {track_id}",(int(x1 + 3), int(y1) - 2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)


    # Display the output frame
    cv2.imshow("Tracking", frame)
    # out.write(frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

