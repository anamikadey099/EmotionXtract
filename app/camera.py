import threading
import cv2
from threading import Lock
from datetime import datetime

class Camera:
    """Camera class to handle webcam operations."""
    def __init__(self):
        self.cap = None  # Camera instance
        self.cap_lock = Lock()
        self.initialized = False  # Track initialization status

    def initialize(self):
        """Safely initialize the camera."""
        with self.cap_lock:
            if not self.initialized:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if self.cap.isOpened():
                    print("Camera initialized successfully.")
                    self.initialized = True
                else:
                    raise Exception("Error: Cannot access the webcam.")

    def start_async_initialization(self):
        """Start camera initialization in a separate thread."""
        init_thread = threading.Thread(target=self.initialize, daemon=True)
        init_thread.start()

    def release(self):
        """Safely release the webcam."""
        with self.cap_lock:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                self.initialized = False
                print("Camera released.")

    def generate_frames(self, emotion_detector, session_data):
        """Generates video frames with detected emotions."""
        self.initialize()  # Ensure the camera is initialized
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            with self.cap_lock:
                ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Convert to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray_frame[y:y + h, x:x + w]
                emotion = emotion_detector.predict(face)

                # Save emotion data
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                session_data.loc[len(session_data)] = {'Timestamp': timestamp, 'Emotion': emotion}

                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode the frame for live streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Failed to encode frame.")
                break

            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')