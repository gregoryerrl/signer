import cv2
import mediapipe as mp
import numpy as np
import json
import time

class SignTester:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.trained_signs = self.load_sign_data()
        self.sentence = []
        self.last_sign_time = time.time()
        self.buffer_time = 2.0  # Buffer time between sign detections
        self.last_detected_sign = None

    def load_sign_data(self):
        """Load trained signs from JSON."""
        try:
            with open("sign_data/signs.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("No trained signs found.")
            return {}

    def extract_landmarks(self, results):
        """Extract landmarks from MediaPipe results."""
        landmarks = {'face': None, 'left_hand': None, 'right_hand': None}
        if results.face_landmarks:
            landmarks['face'] = [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
        if results.left_hand_landmarks:
            landmarks['left_hand'] = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        if results.right_hand_landmarks:
            landmarks['right_hand'] = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        return landmarks

    def compare_landmarks(self, current, trained):
        """Compare current landmarks with trained landmarks."""
        score = 0
        total = 0
        for key in ['left_hand', 'right_hand', 'face']:
            if current[key] and trained.get(key):
                current_arr = np.array(current[key])
                trained_arr = np.array(trained[key])
                if current_arr.shape != trained_arr.shape:
                    continue
                distance = np.mean(np.sqrt(np.sum((current_arr - trained_arr) ** 2, axis=1)))
                similarity = 1 / (1 + distance)
                score += similarity
                total += 1
        return score / total if total > 0 else 0

    def detect_sign(self, results):
        """Detect the sign from current landmarks and return the best match."""
        current_landmarks = self.extract_landmarks(results)
        best_match = None
        best_score = 0.8  # Minimum threshold
        
        for sign_name, sign_data in self.trained_signs.items():
            score = self.compare_landmarks(current_landmarks, sign_data)
            if score > best_score:
                best_score = score
                best_match = sign_name
        
        # Use buffer time to prevent rapid detection
        if best_match and (time.time() - self.last_sign_time > self.buffer_time):
            self.last_sign_time = time.time()
            return best_match
        return None

    def update_sentence(self, detected_sign):
        """Update the sentence with the detected sign."""
        if detected_sign == "_":
            self.sentence.append(" ")
        elif detected_sign == ">":
            if self.sentence:
                self.sentence.pop()
        else:
            self.sentence.append(detected_sign)

    def draw_panel(self, display):
        """Draw the text panel beside the camera feed."""
        cv2.rectangle(display, (20, 20), (620, 700), (50, 50, 50), -1)
        
        # Display the constructed sentence
        sentence_text = "".join(self.sentence)
        words = sentence_text.split()
        lines = []
        current_line = []

        for word in words:
            if len(" ".join(current_line + [word])) > 25:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(" ".join(current_line))

        # Render each line of the sentence
        y_pos = 60
        for line in lines:
            cv2.putText(display, line, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_pos += 40

        # Display the last detected sign
        if self.last_detected_sign:
            cv2.putText(display, f"Detected Sign: {self.last_detected_sign}", (40, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show buffer progress bar
        time_since_last = time.time() - self.last_sign_time
        if time_since_last < self.buffer_time:
            progress = int((time_since_last / self.buffer_time) * 100)
            cv2.rectangle(display, (40, 650), (40 + progress * 5, 670), (0, 255, 0), -1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Sign Language to Text', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sign Language to Text', 1280, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame and convert to RGB
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)

            # Create a display canvas
            display = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame_resized = cv2.resize(frame, (640, 720))
            display[:, 640:] = frame_resized

            # Detect signs
            detected_sign = self.detect_sign(results)
            if detected_sign:
                self.last_detected_sign = detected_sign
                self.update_sentence(detected_sign)

            # Draw the panel with the text
            self.draw_panel(display)
            
            cv2.imshow('Sign Language to Text', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = SignTester()
    tester.run()
