import cv2
import numpy as np
from keras.models import model_from_json  # type: ignore

class EmotionDetector:
    """EmotionDetector class for loading model and predicting emotions."""
    def __init__(self):
        self.model = self.load_model()
        self.class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                             4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    def load_model(self):
        """Load the pre-trained emotion detection model."""
        try:
            with open('model/facialemotiondetector.json', 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            model.load_weights('model/facialemotiondetector.weights.h5')
            print("Model loaded successfully.")
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def preprocess_face(self, face):
        """Preprocess the cropped face image."""
        face = cv2.resize(face, (48, 48))  # Resize to match the model input
        face = face.astype('float32') / 255.0
        face = np.reshape(face, (1, 48, 48, 1))  # Reshape for the model
        return face

    def predict(self, face):
        """
        Predict emotion for a given face image.

        Args:
            face: Grayscale cropped face image.

        Returns:
            Predicted emotion label.
        """
        try:
            face = self.preprocess_face(face)
            predictions = self.model.predict(face, verbose=0)
            predicted_class = np.argmax(predictions[0])
            return self.class_labels[predicted_class]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "unknown"