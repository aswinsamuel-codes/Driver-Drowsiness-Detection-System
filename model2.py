import cv2
import numpy as np
import time
import pygame
import os
from datetime import datetime
import csv

# Initialize pygame for alarm sound
pygame.mixer.init()

# Sound files path setup
SOUND_DIR = r"D:\MY works\MY_IDEAS\Drowsiness Detection"
LOW_ALERT = os.path.join(SOUND_DIR, 'low_alert.wav')
MED_ALERT = os.path.join(SOUND_DIR, 'med_alert.wav')
HIGH_ALERT = os.path.join(SOUND_DIR, 'high_alert.wav')

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(SOUND_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Constants for drowsiness detection
EYE_AR_CONSEC_FRAMES_LOW = 15    # First level warning
EYE_AR_CONSEC_FRAMES_MED = 30    # Second level warning
EYE_AR_CONSEC_FRAMES_HIGH = 45   # Highest level warning

# Constants for head posture and distraction
HEAD_TILT_THRESHOLD = 15         # Degrees of acceptable head tilt
DISTRACTION_FRAMES = 50          # Frames looking away before alert

# Calibration settings
CALIBRATION_FRAMES = 100         # Frames to calibrate normal eye behavior
calibration_counter = 0
calibration_completed = False
normal_blink_rate = 0            # Will be calculated during calibration

# Initialize counters and state
COUNTER = 0
ALERT_LEVEL = 0                  # 0: No alert, 1: Low, 2: Medium, 3: High
DISTRACTION_COUNTER = 0
BLINKS = 0
last_blink_time = time.time()
blink_rate = 0                   # Blinks per minute

# Data logging
log_file = os.path.join(LOG_DIR, f"drowsiness_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(log_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Alert_Level", "Eyes_Detected", "Blink_Rate", "Head_Position", "Distraction_Level"])

# Load OpenCV's face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Function to play alert sound based on severity level
def play_alert(level):
    pygame.mixer.music.stop()  # Stop any currently playing sound
    
    if level == 1:
        if os.path.exists(LOW_ALERT):
            pygame.mixer.music.load(LOW_ALERT)
            pygame.mixer.music.play()
        else:
            print(f"Warning: Sound file not found at {LOW_ALERT}")
    elif level == 2:
        if os.path.exists(MED_ALERT):
            pygame.mixer.music.load(MED_ALERT)
            pygame.mixer.music.play()
        else:
            print(f"Warning: Sound file not found at {MED_ALERT}")
    elif level == 3:
        if os.path.exists(HIGH_ALERT):
            pygame.mixer.music.load(HIGH_ALERT)
            pygame.mixer.music.play(-1)  # Loop the highest alert
        else:
            print(f"Warning: Sound file not found at {HIGH_ALERT}")

# Function to speak alerts
def speak_alert(level):
    alert_messages = {
        1: "Warning: You appear drowsy",
        2: "Alert: Wake up!",
        3: "DANGER: PULL OVER NOW!"
    }
    # This would normally use text-to-speech, but for simplicity we'll just print
    if level > 0:
        print(alert_messages[level])

# Function to log data
def log_data(alert_level, eyes_detected, blink_rate, head_position, distraction_level):
    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                         alert_level, eyes_detected, blink_rate, head_position, distraction_level])

# Start video capture
cap = cv2.VideoCapture(0)

# For FPS calculation
prev_time = 0
curr_time = 0
start_time = time.time()

# Font and colors
font = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)

print("Starting Driver Drowsiness Detection System")
print(f"Sound files should be located at: {SOUND_DIR}")
print("Calibration will begin when face is detected...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Also check for profile faces (looking away)
    profiles = profile_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Create status display area
    cv2.rectangle(frame, (10, 10), (310, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), font, 0.6, WHITE, 1)
    
    # Day/night detection based on average brightness
    avg_brightness = np.mean(gray)
    time_sensitivity = "Night Mode" if avg_brightness < 100 else "Day Mode"
    cv2.putText(frame, time_sensitivity, (20, 140), font, 0.6, BLUE, 1)
    
    # Elapsed time display
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (180, 35), font, 0.6, WHITE, 1)
    
    if len(faces) == 0 and len(profiles) == 0:
        cv2.putText(frame, "No face detected", (20, 65), font, 0.6, YELLOW, 2)
        # Reduce counters when no face is detected
        COUNTER = max(0, COUNTER - 1)
        if COUNTER == 0:
            ALERT_LEVEL = 0
            pygame.mixer.music.stop()
    else:
        # Combine regular faces and profile faces for better detection
        all_faces = list(faces) + list(profiles)
        head_position = "Forward" if len(faces) > 0 else "Looking Away"
        
        # Track distraction (looking away)
        if head_position == "Looking Away":
            DISTRACTION_COUNTER += 1
            if DISTRACTION_COUNTER >= DISTRACTION_FRAMES:
                cv2.putText(frame, "DISTRACTED: EYES ON ROAD!", (frame.shape[1]//2 - 180, 80), 
                            font, 0.9, YELLOW, 2)
        else:
            DISTRACTION_COUNTER = max(0, DISTRACTION_COUNTER - 2)  # Decrease counter when looking forward
        
        # Display distraction level
        distraction_level = min(100, int(DISTRACTION_COUNTER / DISTRACTION_FRAMES * 100))
        cv2.putText(frame, f"Distraction: {distraction_level}%", (180, 65), font, 0.6, 
                    YELLOW if distraction_level > 50 else WHITE, 1)
        
        # Process the primary detected face
        for (x, y, w, h) in all_faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)
            
            # Region of interest for eyes
            roi_gray = gray[y:int(y+h/2), x:x+w]  # Upper half of face for eye detection
            roi_color = frame[y:int(y+h/2), x:x+w]
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))
            
            # Display eye count
            cv2.putText(frame, f"Eyes detected: {len(eyes)}", (20, 65), font, 0.6, WHITE, 1)
            
            # Calibration logic
            if not calibration_completed:
                if len(eyes) == 2:  # We have clear view of both eyes
                    calibration_counter += 1
                    cv2.putText(frame, f"Calibrating: {calibration_counter}/{CALIBRATION_FRAMES}", 
                                (20, 170), font, 0.7, GREEN, 2)
                    
                    if calibration_counter >= CALIBRATION_FRAMES:
                        normal_blink_rate = BLINKS / (time.time() - start_time) * 60  # blinks per minute
                        calibration_completed = True
                        print(f"Calibration complete. Normal blink rate: {normal_blink_rate:.1f} blinks/min")
                
                # Skip drowsiness detection during calibration
                continue
            
            # Blink detection logic
            if len(eyes) < 2 and len(eyes) > 0:  # Possible blink
                COUNTER += 1
            else:
                if COUNTER >= 3 and COUNTER <= 7:  # Normal blink duration
                    BLINKS += 1
                    current_time = time.time()
                    # Update blink rate (blinks per minute)
                    blink_rate = BLINKS / (current_time - start_time) * 60
                    last_blink_time = current_time
                COUNTER = 0
            
            # Display blink rate
            cv2.putText(frame, f"Blink rate: {blink_rate:.1f}/min", (20, 95), font, 0.6, WHITE, 1)
            
            # Update drowsiness counter based on eye detection
            if len(eyes) < 2:  # Less than 2 eyes detected - potential drowsiness
                COUNTER += 1
                cv2.putText(frame, f"Eyes closed: {COUNTER} frames", (20, 117), font, 0.6, ORANGE, 1)
            else:
                # Reset counter if eyes are detected
                COUNTER = max(0, COUNTER - 2)  # Decrease counter more quickly when eyes are open
                cv2.putText(frame, "Eyes open", (20, 117), font, 0.6, GREEN, 1)
            
            # Adjust thresholds based on time of day
            alert_threshold_modifier = 0.7 if time_sensitivity == "Night Mode" else 1.0
            low_threshold = int(EYE_AR_CONSEC_FRAMES_LOW * alert_threshold_modifier)
            med_threshold = int(EYE_AR_CONSEC_FRAMES_MED * alert_threshold_modifier)
            high_threshold = int(EYE_AR_CONSEC_FRAMES_HIGH * alert_threshold_modifier)
            
            # Determine alert level based on counter
            prev_alert = ALERT_LEVEL
            
            if COUNTER >= high_threshold:
                ALERT_LEVEL = 3
            elif COUNTER >= med_threshold:
                ALERT_LEVEL = 2
            elif COUNTER >= low_threshold:
                ALERT_LEVEL = 1
            else:
                ALERT_LEVEL = 0
            
            # Consider abnormal blink rate as an additional drowsiness factor
            if calibration_completed and blink_rate < normal_blink_rate * 0.6:
                # Significantly lower blink rate than normal
                ALERT_LEVEL = max(ALERT_LEVEL, 1)  # At least level 1 alert
            
            # Play sound and speak alert if level has changed
            if ALERT_LEVEL != prev_alert:
                play_alert(ALERT_LEVEL)
                speak_alert(ALERT_LEVEL)
            
            # Draw bounding box for each eye
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), GREEN, 2)
            
            # Log data every 5 seconds
            if int(time.time()) % 5 == 0:
                log_data(ALERT_LEVEL, len(eyes), blink_rate, head_position, distraction_level)
            
            # Only process first face to avoid duplication
            break
    
    # Display alert based on level
    if ALERT_LEVEL == 1:
        cv2.putText(frame, "DROWSINESS WARNING", (frame.shape[1]//2 - 150, 30), font, 0.8, YELLOW, 2)
    elif ALERT_LEVEL == 2:
        cv2.putText(frame, "DROWSINESS ALERT!", (frame.shape[1]//2 - 140, 30), font, 0.9, ORANGE, 2)
    elif ALERT_LEVEL == 3:
        # Flashing red alert for highest level
        if int(time.time() * 2) % 2 == 0:  # Flash at 2Hz
            cv2.putText(frame, "WAKE UP! DANGER!", (frame.shape[1]//2 - 180, 40), font, 1.2, RED, 3)
            # Add red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), RED, -1)
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    
    # Show the frame
    cv2.imshow("Driver Drowsiness Detection", frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.quit()

print(f"Session ended. Log saved to {log_file}")