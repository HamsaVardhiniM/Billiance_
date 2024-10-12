# # from ultralytics import YOLO
# # model = YOLO(r'C:\Users\ubair\Desktop\workspace\Shoplifting-Detection\best2.pt')

# # model.predict(r'c:\Users\ubair\Downloads\WhatsApp Video 2024-09-15 at 13.03.05_c94fe913.mp4' , save = True , conf = 0.01)

# import cv2
# from ultralytics import YOLO

# # Load the YOLO model
# model = YOLO(r'C:\Users\ubair\Desktop\workspace\Shoplifting-Detection\best2.pt')

# # Path to the video
# video_path = r'C:\Users\ubair\Downloads\WhatsApp Video 2024-09-15 at 11.35.24_80194cc8.mp4'

# # Run the YOLO model to predict
# results = model.predict(source=video_path)

# # Load the video with OpenCV
# cap = cv2.VideoCapture(video_path)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('annotated_output.mp4', fourcc, fps, (width, height))

# # Iterate over frames and results
# for idx, result in enumerate(results):
#     # Read frame from video
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Iterate over detections in the result
#     for box in result.boxes:
#         # Get the bounding box coordinates and confidence score
#         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#         conf = box.conf[0].item()  # Confidence score

#         # Determine the color based on confidence (Red if below 0.20, Blue otherwise)
#         color = (0, 0, 255) if conf < 0.3 else (255, 0, 0)

#         # Draw the bounding box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#         # Optionally, add a confidence label above the bounding box
#         label = f'{conf:.2f}'
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # Write the frame to the output video
#     out.write(frame)

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print("Video with annotations saved as 'annotated_output.mp4'")

from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best2.pt")

# Open the video file
video_path =r"A:\Shoplifting-Detection\Shoplifting-Detection\test2.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history and confidence scores
track_history = defaultdict(lambda: [])
confidence_history = defaultdict(lambda: None)  # To store previous confidence scores

# Confidence drop threshold for detecting shoplifting
CONFIDENCE_DROP_THRESHOLD = 0.3

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes, confidence scores, and track IDs
        boxes = results[0].boxes.xywh.cpu()
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks and detect potential shoplifting
        for box, confidence, track_id in zip(boxes, confidences, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # Retain the last 30 tracks (e.g., 30 frames)
                track.pop(0)

            # Track the confidence score for each object and detect drops
            if confidence_history[track_id] is not None:
                previous_confidence = confidence_history[track_id]
                confidence_drop = previous_confidence - confidence

                # If confidence drops beyond the threshold, flag as shoplifting
                if confidence_drop > CONFIDENCE_DROP_THRESHOLD:
                    color = (0, 0, 255)  # Red for potential shoplifting
                    label = "Shoplifting Detected!"
                else:
                    color = (0, 255, 0)  # Green for normal tracking
                    label = "Tracking"
            else:
                color = (0, 255, 0)  # Initial color if there's no prior confidence
                label = "Tracking"

            # Update the confidence history with the current confidence
            confidence_history[track_id] = confidence

            # Draw the bounding box and label
            cv2.rectangle(annotated_frame, 
                          (int(x - w / 2), int(y - h / 2)), 
                          (int(x + w / 2), int(y + h / 2)), 
                          color, 2)
            cv2.putText(annotated_frame, label, (int(x - w / 2), int(y - h / 2) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
