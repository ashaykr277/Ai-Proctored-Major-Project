import os
import cv2
import face_recognition
import shutil
from tkinter import filedialog

# Function to upload an image
def upload_image(username):
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png")])
    
    if image_path:
        registered_images_dir = "utils/registered_images"
        os.makedirs(registered_images_dir, exist_ok=True)

        new_image_path = os.path.join(registered_images_dir, f"{username}.jpg")
        shutil.copy(image_path, new_image_path)

        return new_image_path
    else:
        return None

# Function to recognize the face
def recognize_face(known_face_path):
    # Load the known face image
    known_image = face_recognition.load_image_file(known_face_path)
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Start capturing video
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check for matches with known face
        face_matched = False
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
            if True in matches:
                face_matched = True
                break

        # Show the frame with the recognized face
        if face_matched:
            cv2.putText(frame, "Face Matched!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face Not Matched!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Webcam', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return face_matched
