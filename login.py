import cv2
import face_recognition
import sqlite3
import os
from utils.database import authenticate_user  # Ensure this import is correct

# Function to recognize face
def recognize_face(known_face_path):
    # Load the known image of the user
    known_image = face_recognition.load_image_file(known_face_path)
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Capture the user's face via webcam
    video_capture = cv2.VideoCapture(0)

    # Check if webcam is opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return False
    
    # Read a frame from the webcam
    ret, frame = video_capture.read()
    
    # Release the video capture object
    video_capture.release()
    
    if not ret:
        print("Error: Failed to capture image.")
        return False

    # Convert the image from BGR to RGB
    rgb_frame = frame[:, :, ::-1]
    
    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Check if any faces match
    face_matched = False
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        if True in matches:
            face_matched = True
            break
    
    return face_matched

# Function to start proctoring
def start_proctoring():
    print("Starting proctoring process...")
    username = input("Enter your username: ").strip()  # Strip spaces
    password = input("Enter your password: ").strip()  # Strip spaces

    if authenticate_user(username, password):
        print("Authentication successful!")

        # Path to the registered user's image
        known_face_path = f"utils/images/{username}.jpg"  # Adjust path as needed

        # Check if the image file exists
        if os.path.exists(known_face_path):
            if recognize_face(known_face_path):
                print("Face matched! Proctoring in progress...")
                continue_proctoring()  # Placeholder for further proctoring steps
            else:
                print("Face did not match. Access denied.")
        else:
            print("No registered face image found for this user.")
    else:
        print("Authentication failed! Invalid username or password.")

# Example function for further proctoring steps
def continue_proctoring():
    print("Implement further proctoring logic here...")  # Placeholder

# Run the proctoring process
if __name__ == "__main__":
    start_proctoring()
