import cv2
import mediapipe as mp
import numpy as np
import json
import time
from autocorrect import Speller
import requests
import pygame
import os
from queue import Queue
from threading import Thread

class TextToSpeechWorker:
    def __init__(self):
        self.queue = Queue()
        self.is_speaking = False
        self.thread = Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        pygame.mixer.init()

    def _process_queue(self):
        while True:
            if not self.is_speaking and not self.queue.empty():
                text = self.queue.get()
                self.is_speaking = True
                self._text_to_speech(text)
                self.is_speaking = False
            time.sleep(0.1)

    def _text_to_speech(self, sentence):
        """Convert text to speech using Play.ht API"""
        try:
            url = "https://api.play.ht/api/v2/tts/stream"
            headers = {
                'X-USER-ID': 'nD3EQXVYNWgHncvsW6Sl5s0eoRF2',
                'AUTHORIZATION': '39f94d3b6ee147b8a637bcb38d1b820d',
                'accept': 'audio/mpeg',
                'content-type': 'application/json'
            }
            data = {
                "text": sentence,
                "voice": "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
                "output_format": "mp3"
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            with open("result.mp3", "wb") as file:
                file.write(response.content)
            
            pygame.mixer.music.load("result.mp3")
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            pygame.mixer.music.unload()
            if os.path.exists("result.mp3"):
                os.remove("result.mp3")
                
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")

    def speak(self, text):
        """Add text to speech queue"""
        self.queue.put(text)

    def is_busy(self):
        """Check if currently speaking"""
        return self.is_speaking

class SignTester:
    def __init__(self):
        # Initialize spell checker and TTS
        self.spell = Speller()
        self.tts = TextToSpeechWorker()
        self.corrected_sentence = ""
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        
        self.trained_signs = self.load_sign_data()
        self.sentence = []
        self.last_sign_time = time.time()
        self.buffer_time = 1.5  # Reduced buffer time for better responsiveness
        self.current_sign = None
        self.sign_history = []  # Keep track of recent signs for better accuracy
        self.history_size = 3

    def load_sign_data(self):
        """Load trained signs from JSON."""
        try:
            with open("sign_data/signs.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("No trained signs found.")
            return {}

    def draw_hand_box(self, frame, landmarks):
        """Draw a bounding box around the hand landmarks."""
        points = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in landmarks.landmark])
        
        # Add padding to the box
        padding = 20
        x_min, x_max = int(min(points[:, 0])) - padding, int(max(points[:, 0])) + padding
        y_min, y_max = int(min(points[:, 1])) - padding, int(max(points[:, 1])) + padding
        
        # Ensure coordinates are within frame bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        # Draw the box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # Draw center point
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)

    def extract_landmarks(self, results):
        """Extract landmarks from MediaPipe results."""
        landmarks = {'left_hand': None, 'right_hand': None}
        
        if not results.multi_hand_landmarks:
            return landmarks
            
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_type = results.multi_handedness[hand_idx].classification[0].label.lower()
            landmarks[f"{hand_type}_hand"] = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            
        return landmarks

    def compare_landmarks(self, current, trained):
        """Compare current landmarks with trained landmarks."""
        score = 0
        total = 0
        
        for key in ['left_hand', 'right_hand']:
            if current[key] and trained.get(key):
                current_arr = np.array(current[key])
                trained_arr = np.array(trained[key])
                if current_arr.shape == trained_arr.shape:
                    distance = np.mean(np.sqrt(np.sum((current_arr - trained_arr) ** 2, axis=1)))
                    similarity = 1 / (1 + distance)
                    score += similarity
                    total += 1
                    
        return score / total if total > 0 else 0

    def detect_sign(self, results):
        """Detect the sign from current landmarks with buffering and history."""
        current_landmarks = self.extract_landmarks(results)
        if not current_landmarks['left_hand'] and not current_landmarks['right_hand']:
            self.current_sign = None
            return None
    
        best_match = None
        best_score = 0.7  # Threshold for detection
        
        for sign_name, sign_data in self.trained_signs.items():
            score = self.compare_landmarks(current_landmarks, sign_data)
            if score > best_score:
                best_score = score
                best_match = sign_name
        
        # Update current sign for real-time display
        self.current_sign = best_match
        
        # Use sign history for more stable detection
        if best_match:
            self.sign_history.append(best_match)
            if len(self.sign_history) > self.history_size:
                self.sign_history.pop(0)
            
            # Only return sign if it appears consistently in history
            if len(self.sign_history) == self.history_size and \
               all(sign == self.sign_history[0] for sign in self.sign_history) and \
               time.time() - self.last_sign_time > self.buffer_time:
                self.last_sign_time = time.time()
                return best_match
            
        return None

    def update_sentence(self, detected_sign):
        """Update the sentence based on detected sign."""
        if not detected_sign:
            return

        if detected_sign == "_":  # Space
            if self.sentence and self.sentence[-1] != " ":
                self.sentence.append(" ")
        elif detected_sign == ">":  # Backspace
            if self.sentence:
                self.sentence.pop()
        elif detected_sign == "-" and not self.tts.is_busy():  # Enter/Speak
            sentence_text = "".join(self.sentence).strip()
            if sentence_text:
                # Apply spell check to each word
                words = sentence_text.split()
                corrected_words = [self.spell(word) for word in words]
                self.corrected_sentence = " ".join(corrected_words)
                
                # Speak the corrected sentence
                self.tts.speak(self.corrected_sentence)
                
                # Reset after speaking
                self.sentence.clear()
                self.corrected_sentence = ""
        else:
            self.sentence.append(detected_sign)
            # Update spell-checked sentence
            sentence_text = "".join(self.sentence)
            words = sentence_text.split()
            self.corrected_sentence = " ".join(self.spell(word) for word in words)

    def draw_panel(self, frame):
        """Draw the UI panel with current sentence and sign information."""
        # Create semi-transparent overlay for the top panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Display raw sentence
        raw_text = "".join(self.sentence)
        cv2.putText(frame, "Current: " + raw_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display corrected sentence
        cv2.putText(frame, "Corrected: " + self.corrected_sentence, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display current sign
        sign_text = self.current_sign if self.current_sign else "No sign detected"
        cv2.putText(frame, "Current Sign: " + sign_text, (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # Display controls
        controls = "Controls: '_' for space, '>' for backspace, '-' for speak"
        cv2.putText(frame, controls, (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Display speaking status
        if self.tts.is_busy():
            cv2.putText(frame, "Speaking...", (frame.shape[1]-150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw buffer progress bar
        time_since_last = time.time() - self.last_sign_time
        if time_since_last < self.buffer_time:
            progress = int((time_since_last / self.buffer_time) * 200)
            cv2.rectangle(frame, (20, 190), (20 + progress, 200), (0, 255, 0), -1)

    def run(self):
        """Main loop for sign detection and sentence building."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\nLoaded signs:", ', '.join(sorted(self.trained_signs.keys())))
        print("\nControls:")
        print("'_' - Add space")
        print("'>' - Backspace")
        print("'-' - Speak sentence")
        print("'q' - Quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror for intuitive interaction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Draw hand landmarks and boxes
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )
                    # Draw bounding box
                    self.draw_hand_box(frame, hand_landmarks)

            # Detect and process signs
            detected_sign = self.detect_sign(results)
            if detected_sign:
                self.update_sentence(detected_sign)

            # Draw UI
            self.draw_panel(frame)
            
            cv2.imshow('Sign Language to Text', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = SignTester()
    tester.run()

