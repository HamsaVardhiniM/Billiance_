import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing for visualization
mp_drawing = mp.solutions.drawing_utils

# Open webcam or video file
cap = cv2.VideoCapture(r"c:\Users\ubair\Downloads\WhatsApp Video 2024-09-15 at 11.35.24_80194cc8.mp4")  # Video file capture
 # Change to video file path if necessary

def detect_shoplifting(pose_landmarks):
    # Logic for detecting potential shoplifting activity
    # Using landmarks like wrists, hips, knees, and ankles 
    left_wrist = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    
    # Example detection: Check if both wrists move near the waist/hip area (could indicate hiding an item)
    if (left_wrist.y > left_hip.y) and (right_wrist.y > right_hip.y):
        return True  # Suspicious activity detected (e.g., hiding something)
    
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame color to RGB as MediaPipe uses RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect poses
    results = pose.process(image)

    # If poses are detected
    if results.pose_landmarks:
        # Draw the pose annotation on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Detect shoplifting behavior
        if detect_shoplifting(results.pose_landmarks.landmark):
            cv2.putText(frame, "Shoplifting Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Pose Estimation', frame)

    # Exit on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
