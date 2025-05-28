import cv2
import torch
from pathlib import Path
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load COCO names
coco_names = Path('coco.names').read_text().strip().split('\n')

# Define the classes of interest
obstacle_classes = ['person', 'chair', 'table', 'rock', 'fire hydrant', 'sofa']

# Initialize video capture
cap = cv2.VideoCapture(0) # Use 0 for webcam, or replace with video file path

# Variables to track the state
clear_path = True
obstacle_detected = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Assume clear path until proven otherwise
    clear_path = True

    # Draw bounding boxes and labels on the frame for obstacle classes
    for *xyxy, conf, cls in results.xyxy[0]:
        # Only process objects with confidence greater than 0.5
        if conf > 0.5:
            x1, y1, x2, y2 = map(int, xyxy) # Convert to int for drawing
            label = coco_names[int(cls)]
            if label in obstacle_classes: # Check if the detected object is an obstacle
                color = (0, 0, 0) # Black color for obstacles
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # If obstacle detected and clear_path was previously True, announce the obstacle
                if not obstacle_detected:
                    engine.say(f"{label} ahead")
                    engine.runAndWait()
                    obstacle_detected = True
                clear_path = False

    # If no obstacles are detected in the current frame, reset the obstacle_detected flag
    if clear_path:
        obstacle_detected = False

    # Display the resulting frame
    cv2.imshow('Obstacle Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
