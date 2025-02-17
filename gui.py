import os
import cv2
import face_recognition
import shutil
import threading
import numpy as np
from tkinter import filedialog, messagebox, Button, Label, Entry, Frame, Tk

# Define file path for storing user credentials
USER_DATA_FILE = "users.txt"

# Load YOLO model for object detection
net = cv2.dnn.readNet("utils/yolov3.weights", "utils/yolov3.cfg")  # Update paths based on your structure
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
class_names = []
with open("utils/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to register a new user
def register_user(username, password):
    if username_exists(username):
        return False
    with open(USER_DATA_FILE, "a") as user_file:
        user_file.write(f"{username},{password}\n")
    return True

def username_exists(username):
    if not os.path.exists(USER_DATA_FILE):
        return False
    with open(USER_DATA_FILE, "r") as user_file:
        for line in user_file:
            if line.strip().split(",")[0] == username:
                return True
    return False

def authenticate_user(username, password):
    if not os.path.exists(USER_DATA_FILE):
        return False
    with open(USER_DATA_FILE, "r") as user_file:
        for line in user_file:
            stored_username, stored_password = line.strip().split(",")
            if stored_username == username and stored_password == password:
                return True
    return False

def upload_image(username):
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", ".jpg;.png")])
    if image_path:
        registered_images_dir = "utils/registered_images"
        os.makedirs(registered_images_dir, exist_ok=True)
        new_image_path = os.path.join(registered_images_dir, f"{username}.jpg")
        shutil.copy(image_path, new_image_path)
        return new_image_path
    else:
        return None

def recognize_face(known_face_path):
    known_image = face_recognition.load_image_file(known_face_path)
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Start capturing video
    video_capture = cv2.VideoCapture(0)

    # Capture frames for 5 seconds
    start_time = cv2.getTickCount()
    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Show bounding boxes for detected faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
            if True in matches:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Face Matched!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "No Match!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Webcam', frame)

        # Check if 5 seconds have passed
        if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > 5:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return face_locations, face_encodings

class ProctoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Proctored Exam System")
        self.root.geometry("300x300")

        self.frame = Frame(self.root)
        self.frame.pack(pady=20)

        self.welcome_label = Label(self.frame, text="Welcome to AI Proctored Examination System")
        self.welcome_label.pack()

        self.register_button = Button(self.frame, text="Register", command=self.open_register_window)
        self.register_button.pack(pady=10)

        self.login_button = Button(self.frame, text="Login", command=self.open_login_window)
        self.login_button.pack(pady=10)

    def open_register_window(self):
        self.register_window = Tk()
        self.register_window.title("Register")

        Label(self.register_window, text="Username").pack()
        self.username_entry = Entry(self.register_window)
        self.username_entry.pack()

        Label(self.register_window, text="Password").pack()
        self.password_entry = Entry(self.register_window, show="*")
        self.password_entry.pack()

        Button(self.register_window, text="Upload Image", command=self.upload_image).pack()
        Button(self.register_window, text="Register", command=self.register).pack()

    def upload_image(self):
        username = self.username_entry.get()
        if username:
            self.image_path = upload_image(username)
        else:
            messagebox.showwarning("Warning", "Please enter a username before uploading an image.")

    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if register_user(username, password):
            messagebox.showinfo("Success", "User registered successfully!")
            self.register_window.destroy()
        else:
            messagebox.showerror("Error", "Username already exists!")

    def open_login_window(self):
        self.login_window = Tk()
        self.login_window.title("Login")

        Label(self.login_window, text="Username").pack()
        self.login_username_entry = Entry(self.login_window)
        self.login_username_entry.pack()

        Label(self.login_window, text="Password").pack()
        self.login_password_entry = Entry(self.login_window, show="*")
        self.login_password_entry.pack()

        Button(self.login_window, text="Login", command=self.login).pack()

    def login(self):
        username = self.login_username_entry.get()
        password = self.login_password_entry.get()

        if authenticate_user(username, password):
            messagebox.showinfo("Success", "Authentication successful!")
            self.login_window.destroy()
            self.start_proctoring(username)  # Start proctoring after successful login
        else:
            messagebox.showerror("Error", "Authentication failed! User not found.")

    def start_proctoring(self, username):
        known_face_path = f"utils/registered_images/{username}.jpg"
        if os.path.exists(known_face_path):
            face_locations, face_encodings = recognize_face(known_face_path)
            if face_encodings:  # Check if any face is detected
                messagebox.showinfo("Success", "Face matched!")
                self.webcam_monitoring(username, face_locations)  # Pass face_locations for monitoring
            else:
                messagebox.showerror("Error", "Face did not match. Access denied.")
        else:
            messagebox.showerror("Error", "No registered face image found for this user.")

    def webcam_monitoring(self, username, initial_face_locations):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height
        cap.set(cv2.CAP_PROP_FPS, 15)  # Set a lower frame rate (15 FPS)
        
        suspicious_activity_count = 0
        limit = 10

        # Thread for processing frames
        def process_frames(face_locations):
            nonlocal suspicious_activity_count

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Object detection
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []

                # Process outputs
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:  # Confidence threshold
                            center_x = int(detection[0] * frame.shape[1])
                            center_y = int(detection[1] * frame.shape[0])
                            w = int(detection[2] * frame.shape[1])
                            h = int(detection[3] * frame.shape[0])

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)  # Non-max suppression

                # Face detection using face_recognition
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for i in indexes.flatten():
                    box = boxes[i]
                    (x, y, w, h) = box
                    label = str(class_names[class_ids[i]])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Detect cell phone
                    if label == "cell phone":
                        suspicious_activity_count += 1
                        self.log_suspicious_activity(username, "Cell phone detected")
                        cv2.putText(frame, "Phone Detected!", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check for multiple faces detected
                if len(face_locations) > 1:
                    suspicious_activity_count += 1
                    self.log_suspicious_activity(username, "Multiple faces detected")
                    cv2.putText(frame, "Multiple Faces Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Log out if suspicious activity limit reached
                if suspicious_activity_count >= limit:
                    self.log_out(username)
                    break

                cv2.imshow("Proctoring", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        # Start the frame processing in a separate thread
        threading.Thread(target=process_frames, args=(initial_face_locations,)).start()

    def log_suspicious_activity(self, username, message):
        with open("malicious_activity_log.txt", "a") as log_file:
            log_file.write(f"{username}: {message}\n")

    def log_out(self, username):
        messagebox.showwarning("Warning", "Suspicious activity limit reached! Logging out...")
        # Implement your logout logic here
        self.root.quit()  # Close the application or redirect as needed

if __name__ == "__main__":
    root = Tk()
    app = ProctoringApp(root)
    root.mainloop()