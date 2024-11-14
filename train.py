import cv2
import mediapipe as mp
import numpy as np
import json
import os

class SignRecorder:
    def __init__(self):
        # Initialize MediaPipe Holistic model
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.data_file = "sign_data/signs.json"
        self.signs_data = self.load_sign_data()

    def load_sign_data(self):
        """Load existing sign data from JSON file."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {}

    def save_sign_data(self):
        """Save the collected sign data to JSON."""
        with open(self.data_file, 'w') as f:
            json.dump(self.signs_data, f, indent=4)
        print(f"Data saved to {self.data_file}")

    def capture_landmarks(self, results):
        """Extract and normalize landmarks from MediaPipe results."""
        landmarks = {'face': None, 'left_hand': None, 'right_hand': None}

        # Extract face landmarks
        if results.face_landmarks:
            landmarks['face'] = [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]

        # Extract left and right hand landmarks
        if results.left_hand_landmarks:
            landmarks['left_hand'] = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]

        if results.right_hand_landmarks:
            landmarks['right_hand'] = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            # Mirror right hand to create left hand
            landmarks['left_hand'] = self.mirror_right_hand(landmarks['right_hand'])

        return landmarks

    def mirror_right_hand(self, right_hand_landmarks):
        """Mirror right hand landmarks to create left hand landmarks."""
        if not right_hand_landmarks:
            return None
        # Invert the x-coordinate to mirror the hand
        return [[1 - lm[0], lm[1], lm[2]] for lm in right_hand_landmarks]

    def add_sign(self, sign_name, landmarks):
        """Add a new sign to the dataset."""
        self.signs_data[sign_name] = landmarks
        print(f"Added sign: {sign_name}")

    def run(self):
        cap = cv2.VideoCapture(0)
        print("Press 's' to save a sign and 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)

            # Display landmarks
            if results.face_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

            cv2.imshow('Training', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                sign_name = input("Enter sign name: ")
                landmarks = self.capture_landmarks(results)
                self.add_sign(sign_name, landmarks)
                self.save_sign_data()
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recorder = SignRecorder()
    recorder.run()
