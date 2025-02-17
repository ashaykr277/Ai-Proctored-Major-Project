import cv2
import face_recognition
import numpy as np

# Initialize an empty list to store known face encodings and their names
known_face_encodings = []
known_face_names = []

def add_known_face(name, image_path):
    """Adds a known face to the known faces list."""
    try:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    except Exception as e:
        print(f"Error loading image: {e}")

def detect_faces(frame):
    """Detects faces in the frame and returns the frame with face rectangles drawn and the count of unique faces."""
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_count = 0
    recognized_faces = set()  # To store unique recognized face encodings

    for face_encoding in face_encodings:
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if True in matches:
            face_count += 1  # Count this unique face
            recognized_faces.add(tuple(face_encoding))  # Store the unique face encoding

    # Draw rectangles around detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    return frame, face_count
