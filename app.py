import streamlit as st
from scipy.spatial import distance as dist
from imutils import face_utils
from twilio.rest import Client
import numpy as np
import imutils
import dlib
import cv2
import pyttsx3
import threading
import time
import pygame

# Twilio configuration for sending emergency SMS alerts
  # Replace with your Twilio Auth Token
account_sid = st.secrets.Twilio.sid  # Replace with your Twilio Account SID
auth_token = st.secrets.Twilio.token
twilio_client = Client(account_sid, auth_token)
twilio_number = st.secrets.Twilio.number
  # Replace with your Twilio phone number

# Streamlit input field for entering the emergency contact number
st.title("Drowsiness Detection System with Emergency Alert")
emergency_contact = st.text_input("Enter Emergency Contact Number (with country code)", value="+919306292328")

# Set up the Streamlit layout with title
st.title("Drowsiness and Yawning Detection")

# Load dlib's face detector and the facial landmark predictor for face analysis
predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Path to pre-trained dlib model for facial landmarks
detector = dlib.get_frontal_face_detector()  # Initialize face detector
predictor = dlib.shape_predictor(predictor_path)  # Initialize predictor for 68 face landmarks

# Constants and thresholds for detection
EYE_AR_THRESH = 0.25  # EAR threshold below which eyes are considered closed
MOU_AR_THRESH = 0.75  # MAR threshold above which mouth is considered open (yawning)
YAWN_LIMIT = 2  # Maximum number of yawns allowed before warning
MAX_WARNINGS = 1  # Maximum warnings allowed before emergency alert
EYE_CLOSED_THRESHOLD_SEC = 4  # Duration (in seconds) eyes must be closed to trigger alert

# Variables for tracking drowsiness state
yawnStatus = False
yawns = 0
warning_count = 0
tts_active = False
eye_closed_start_time = None

# Placeholders for displaying video and status in Streamlit
frame_placeholder = st.empty()
status_placeholder = st.empty()

# Initialize text-to-speech (TTS) engine
engine = pyttsx3.init()

# Initialize pygame for alarm sound playback
pygame.mixer.init()

# Helper function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance between points 1 and 5
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance between points 2 and 4
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance between points 0 and 3
    return (A + B) / (2.0 * C)  # Calculate EAR as a ratio of vertical to horizontal distances

# Helper function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mou):
    X = dist.euclidean(mou[0], mou[6])  # Horizontal distance between mouth corners
    Y1 = dist.euclidean(mou[2], mou[10])  # Vertical distance between upper and lower mouth points
    Y2 = dist.euclidean(mou[4], mou[8])  # Another vertical distance in the mouth region
    return (Y1 + Y2) / (2.0 * X)  # Calculate MAR as a ratio of vertical to horizontal distances

# Function to play a TTS alert message
def play_tts_alert(text):
    global tts_active
    tts_active = True  # Set TTS as active to prevent overlap
    engine.say(text)  # Load the alert message into TTS engine
    engine.runAndWait()  # Start speaking the alert
    tts_active = False  # Reset TTS active status after completion

# Function to play alarm sound using Pygame
def play_alarm():
    pygame.mixer.music.load('alarm.mp3')  # Load the alarm audio file
    pygame.mixer.music.play()  # Play the alarm sound
    while pygame.mixer.music.get_busy():  # Keep checking if alarm is still playing
        pygame.time.Clock().tick(5)  # Polling interval to avoid excessive CPU usage

# Function to send an SMS alert via Twilio
def send_emergency_sms():
    if emergency_contact:  # Check if emergency contact is provided
        message = twilio_client.messages.create(
            body="Warning: The driver has exceeded the drowsiness threshold. Immediate action is recommended.",
            from_=twilio_number,
            to=emergency_contact
        )  # Send SMS message
        st.write(f"Emergency SMS sent to {emergency_contact}")  # Display confirmation in Streamlit

# Buttons in Streamlit to start and stop detection
start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

if start_button:  # Check if Start button is clicked
    camera = cv2.VideoCapture(0)  # Open default camera for capturing video
    while True:  # Loop to continuously capture frames
        ret, frame = camera.read()  # Read frame from the camera
        if not ret:  # Break if frame capture fails
            break
        frame = imutils.resize(frame, width=640)  # Resize frame for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale for face detection
        prev_yawn_status = yawnStatus  # Store previous yawn status for tracking
        rects = detector(gray, 0)  # Detect faces in the frame

        # Process detections for each face
        for rect in rects:
            shape = predictor(gray, rect)  # Predict facial landmarks for the face
            shape = face_utils.shape_to_np(shape)  # Convert landmarks to NumPy array
            # Extract eye and mouth regions from facial landmarks
            leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
            rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
            mouth = shape[face_utils.FACIAL_LANDMARKS_IDXS["mouth"][0]:face_utils.FACIAL_LANDMARKS_IDXS["mouth"][1]]
            
            # Calculate EAR for each eye and MAR for mouth
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mouEAR = mouth_aspect_ratio(mouth)
            ear = (leftEAR + rightEAR) / 2.0  # Average EAR of both eyes

            # Draw contours around eyes and mouth
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

            # Detect drowsiness if eyes are closed for more than 4 seconds
            if ear < EYE_AR_THRESH:
                if eye_closed_start_time is None:  # Start timing if eyes are closed
                    eye_closed_start_time = time.time()
                else:
                    elapsed_time = time.time() - eye_closed_start_time
                    if elapsed_time >= EYE_CLOSED_THRESHOLD_SEC:
                        if not tts_active:
                            threading.Thread(target=play_tts_alert, args=("Eyes closed for too long. Please stay alert!",)).start()
                        warning_count += 1  # Increment warning count
                        status_placeholder.text("EYES CLOSED WARNING!")  # Display warning message
                        eye_closed_start_time = None  # Reset eye closed timer
            else:
                eye_closed_start_time = None  # Reset timer if eyes open

            # Yawning detection based on mouth openness
            if mouEAR > MOU_AR_THRESH:
                yawnStatus = True
            else:
                yawnStatus = False

            # Count yawns and issue a warning if the limit is reached
            if prev_yawn_status and not yawnStatus:  # Detect end of yawn
                yawns += 1  # Increment yawn count
            if yawns > YAWN_LIMIT:
                yawns = 0  # Reset yawn count
                warning_count += 1  # Increment warning count
                if not tts_active:
                    threading.Thread(target=play_tts_alert, args=("Yawn limit exceeded. Please take a break.",)).start()

            # Check if max warnings exceeded and trigger emergency alert
            if warning_count >= MAX_WARNINGS:
                warning_count = 0  # Reset warning count
                threading.Thread(target=play_alarm).start()  # Play alarm sound
                threading.Thread(target=play_tts_alert, args=("You are not allowed to drive. The car will shut down in 5 minutes if you don't take a break.",)).start()
                send_emergency_sms()  # Send SMS alert

            # Update status message with current yawn and warning counts
            status_placeholder.text(f"Yawn Count: {yawns} | Warnings: {warning_count}")

        # Stream video frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

        # Stop detection if Stop button is clicked
        if stop_button:
            break

    camera.release()  # Release camera resource
    cv2.destroyAllWindows()  # Close OpenCV windows
