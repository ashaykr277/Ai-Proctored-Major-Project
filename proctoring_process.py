import cv2
import face_recognition
import time

# Function to recognize the face with preview
def recognize_face_with_preview(known_face_path, callback):
    # Load the known face image
    known_image = face_recognition.load_image_file(known_face_path)
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Start capturing video
    video_capture = cv2.VideoCapture(0)

    # Set a smaller frame size (e.g., 640x480)
    video_capture.set(3, 640)  # Width
    video_capture.set(4, 480)  # Height

    face_matched = False
    start_time = time.time()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

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

        # Display appropriate message on the frame
        if face_matched:
            cv2.putText(frame, "Face Matched!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected! Try again!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the current frame
        cv2.imshow('Webcam', frame)

        # Break the loop after 5 seconds if face is matched
        if face_matched and (time.time() - start_time > 5):
            break

        # Check for key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    callback(face_matched)  # Callback to indicate face match result