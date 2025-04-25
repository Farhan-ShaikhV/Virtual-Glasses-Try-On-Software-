import cv2
import cvzone
import numpy as np
import mediapipe as mp
from time import time

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Glasses selection
num = 1
overlay = cv2.imread(f'Glasses/glass{num}.png', cv2.IMREAD_UNCHANGED)  # Load first glasses

# UI Elements
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 255, 255)

# Gesture detection variables
COOLDOWN_TIME = 2  # Increased cooldown time to slow down switching
last_gesture_time = 0

def count_fingers(hand_landmarks):
    """Count the number of extended fingers."""
    finger_tips = [8, 12, 16, 20]  # Tip landmarks for index, middle, ring, and pinky fingers
    count = 0
    
    # Get landmark positions
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    
    # Check fingers (if tip is above the corresponding lower joint, it's extended)
    for tip in finger_tips:
        if landmarks[tip][1] < landmarks[tip - 2][1]:  # Compare with PIP joint
            count += 1

    return count  # Returns the number of extended fingers

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Flip the frame for mirror effect
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Add glasses overlay if available
        if overlay is not None:
            overlay_resize = cv2.resize(overlay, (w, int(h * 0.8)))
            frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])

    # Hand gesture detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the number of extended fingers
            num_fingers = count_fingers(hand_landmarks)

            # Detect gestures
            if time() - last_gesture_time > COOLDOWN_TIME:
                if num_fingers == 1 and num < 29:  # One Finger (Next glasses)
                    num += 1
                    overlay = cv2.imread(f'Glasses/glass{num}.png', cv2.IMREAD_UNCHANGED)
                    print(f"Next Glasses: {num}")
                    last_gesture_time = time()

                elif num_fingers == 2 and num > 1:  # Two Fingers (Previous glasses)
                    num -= 1
                    overlay = cv2.imread(f'Glasses/glass{num}.png', cv2.IMREAD_UNCHANGED)
                    print(f"Previous Glasses: {num}")
                    last_gesture_time = time()

    # Add UI elements
    cv2.putText(frame, f"Glasses {num}/29", (20, 40), FONT, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "1 Finger: Next | 2 Fingers: Previous", (20, frame.shape[0] - 50), FONT, 0.6, TEXT_COLOR, 2)
    cv2.putText(frame, "Press 'Q' to Quit", (20, frame.shape[0] - 20), FONT, 0.6, TEXT_COLOR, 2)

    cv2.imshow('SnapLens - Virtual Glasses Try-On', frame)

    # Key handling
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
