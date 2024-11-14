import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import re

class HolisticGestureRecognizer:
    def __init__(self, data_path, confidence_threshold=0.6):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Load gesture data
        self.gesture_data = self._load_gesture_data(data_path)
        self.word = []
        self.last_gesture = None
        self.last_time = time.time()
        
        # Initialize autocorrect buffer
        self.current_input = ""
        self.corrected_text = ""

    def _load_gesture_data(self, data_path):
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
            return {gesture: np.array(sample) for gesture, sample in data.items()}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading gesture data: {e}")
            return {}

    def _get_gesture(self, results):
        """Identify the gesture based on landmarks."""
        if not results.multi_hand_landmarks:
            return None

        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Normalize landmarks for scale invariance
            landmarks -= landmarks[0]

            for gesture, sample_landmarks in self.gesture_data.items():
                diff = np.linalg.norm(landmarks - sample_landmarks, axis=1).mean()
                
                # Adjusted threshold for better detection
                if diff < 0.05:  # Experiment with this value based on your data
                    return gesture
        return None

    def autocorrect(self, input_text):
        """Comprehensive autocorrection system for simple conversations"""
        input_text = input_text.strip().title()
        
        # Static corrections
        static_phrases = {
            "Thank You": "Thank you",
            "I Love You": "I love you",
            "Yes": "Yes",
            "No": "No",
            "Good Meet You": "Nice to meet you",
            "Hi Good Meet You": "Hi, nice to meet you",
            "Me Student": "I am a student",
            "You Student": "Are you a student?"
        }
        
        # Check for static phrases first
        if input_text in static_phrases:
            return static_phrases[input_text]
        
        # Regular expression patterns for dynamic corrections
        patterns = {
            r'^Me Name ([A-Z][a-zA-Z]*)$': lambda m: f"My name is {m.group(1)}",
            r'^Me Live ([A-Z][a-zA-Z\s]*)$': lambda m: f"I live in {m.group(1)}",
            r'^Me (Feel|Am) (Good|Fine|Great|Beautiful)$': lambda m: f"I am {m.group(2).lower()}",
            r'^You (Good|Fine|Great|Beautiful)$': lambda m: f"Are you {m.group(1).lower()}?",
            r'^Me Hungry$': lambda m: "I am hungry",
            r'^You Hungry$': lambda m: "Are you hungry?"
        }
        
        for pattern, replacement in patterns.items():
            match = re.match(pattern, input_text)
            if match:
                return replacement(match)
        
        return input_text

    def process_frame(self, frame):
        """Process a frame and handle word-building."""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)

        current_time = time.time()
        gesture = self._get_gesture(results)

        if gesture and current_time - self.last_time > 0.8:
            if gesture == "_":  # Space gesture
                self.word.append(" ")
            elif gesture == ">":  # Backspace gesture
                if self.word:
                    self.word.pop()
            else:
                self.word.append(gesture)
            
            self.current_input = ''.join(self.word).strip()
            self.corrected_text = self.autocorrect(self.current_input)
            
            self.last_gesture = gesture
            self.last_time = current_time

        return frame

    def get_display_text(self):
        return self.current_input, self.corrected_text

def main():
    cap = cv2.VideoCapture(0)
    recognizer = HolisticGestureRecognizer(data_path="asl_data/holistic_data.json")
    window_name = "ASL Recognition with Autocorrect"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = recognizer.process_frame(frame)
        input_text, corrected_text = recognizer.get_display_text()
        
        height, width = frame.shape[:2]
        output_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
        output_frame[:, :width] = frame

        right_panel = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(right_panel, "Input: " + input_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(right_panel, "Autocorrect: " + corrected_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        output_frame[:, width:] = right_panel
        cv2.line(output_frame, (width, 0), (width, height), (255, 255, 255), 2)

        cv2.imshow(window_name, output_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
