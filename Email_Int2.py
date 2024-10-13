from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
import pickle
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from mimetypes import guess_type as guess_mime_type  # Use this alias for guess_type
import zipfile

# Load the YOLOv8 model
model = YOLO(r"A:\Shoplifting-Detection\Shoplifting-Detection\best2.pt")

# Open the video file
video_path = r"A:\Shoplifting-Detection\Shoplifting-Detection\Shoplifting (7).mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties for saving
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save output
output_path = "Test1_output_annotated_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Store the track history and confidence scores
track_history = defaultdict(lambda: [])
confidence_history = defaultdict(lambda: None)  # To store previous confidence scores

# Confidence drop threshold for detecting shoplifting
CONFIDENCE_DROP_THRESHOLD = 0.3

# Slow-motion factor (e.g., 3x slower)
SLOW_MOTION_FACTOR = 6

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

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(boxes)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Flag to indicate if shoplifting is detected in this frame
        shoplifting_detected = False

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
                    shoplifting_detected = True  # Mark this frame for slow-motion
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

        # If shoplifting is detected, write the frame multiple times to create a slow-motion effect
        if shoplifting_detected:
            for _ in range(SLOW_MOTION_FACTOR):  # Write the frame SLOW_MOTION_FACTOR times
                out.write(annotated_frame)
        else:
            # Write the annotated frame to the output video
            out.write(annotated_frame)

        # Display the annotated frame (optional)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video with slow-motion saved to {output_path}")

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
OUR_EMAIL = 'hamsavardhini.m@gmail.com'  # Replace with your email

def gmail_authenticate():
    """Authenticate and create a service for Gmail API."""
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('Billiance_Gmail_API_Cred.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def create_message_with_attachment(to, subject, body, file_path):
    """Create an email message with an attachment."""
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = OUR_EMAIL
    message['subject'] = subject

    msg_body = MIMEText(body)
    message.attach(msg_body)

    content_type, encoding = guess_mime_type(file_path)  # Use guess_mime_type here
    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    main_type, sub_type = content_type.split('/', 1)

    with open(file_path, 'rb') as f:
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(f.read())
    
    filename = os.path.basename(file_path)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)
    
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw_message}

def send_message(service, to, subject, body, file_path):
    """Send an email message with an attachment."""
    message = create_message_with_attachment(to, subject, body, file_path)
    try:
        message = service.users().messages().send(userId='me', body=message).execute()
        print(f'Sent message to {to} Message Id: {message["id"]}')
    except Exception as error:
        print(f'An error occurred: {error}')

def send_email():
    # Check if the video file exists
    if os.path.exists(output_path):
        # Define the zip file name
        zip_file_path = output_path.replace('.mp4', '.zip')
        
        # Create a zip file with the video file
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(output_path, os.path.basename(output_path))

        # Authenticate and send the email
        service = gmail_authenticate()
        send_message(service, "hamsavardhini.m@gmail.com", "Shoplifting Detected!!", 
                    "Please find the video attached.", zip_file_path)

        # Clean up the zip file after sending
        os.remove(zip_file_path)
    else:
        print("Error: No video file to send.")

if __name__ == "__main__":
    # Call the send_email function automatically after video processing
    send_email()
