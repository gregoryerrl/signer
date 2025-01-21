import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime
import pygame
from queue import Queue
from threading import Thread
from sign_autocorrect import autocorrect

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
        """Convert text to speech using pygame's built-in speech"""
        try:
            # Save text to temporary file
            with open("temp.txt", "w") as f:
                f.write(sentence)
            
            # Use system text-to-speech (platform independent)
            if os.name == 'nt':  # Windows
                os.system(f'powershell -c "Get-Content temp.txt | Out-File -Encoding ASCII temp.aiff; Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak([System.IO.File]::ReadAllText(\'temp.txt\'))"')
            else:  # macOS/Linux
                os.system('say -f temp.txt')
            
            # Cleanup
            if os.path.exists("temp.txt"):
                os.remove("temp.txt")
                
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
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        
        # Load trained signs
        self.data_file = "sign_data/signs.json"
        self.signs_data = self.load_sign_data()
        
        # Initialize TTS
        self.tts = TextToSpeechWorker()
        
        # UI and sentence building
        self.sentence = []
        self.current_sign = None
        self.last_sign_time = time.time()
        self.buffer_time = 2.0  # 3 seconds buffer time
        self.sign_history = []
        self.history_size = 10  # Reduced history size since we're using time-based buffer
        self.detection_start_time = None  # Track when detection starts

    def load_sign_data(self):
        """Load trained signs from JSON file."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {}

    def capture_landmarks(self, results):
        """Extract and normalize landmarks from MediaPipe results."""
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
        """Detect the sign from current landmarks."""
        current_landmarks = self.capture_landmarks(results)
        if not current_landmarks['left_hand'] and not current_landmarks['right_hand']:
            self.current_sign = None
            self.sign_history.clear()  # Clear history when hands disappear
            self.detection_start_time = None  # Reset detection timer
            return None
    
        best_match = None
        best_score = 0.7  # Threshold for detection
        
        for sign_name, sign_data in self.signs_data.items():
            score = self.compare_landmarks(current_landmarks, sign_data)
            if score > best_score:
                best_score = score
                best_match = sign_name
        
        # Update current sign for display purposes
        if best_match != self.current_sign:
            self.sign_history.clear()  # Reset history when sign changes
            self.detection_start_time = None  # Reset detection timer
            
        self.current_sign = best_match
        
        # Use sign history for more stable detection
        if best_match:
            # Start timing when we first detect a sign
            if not self.detection_start_time:
                self.detection_start_time = time.time()
            
            self.sign_history.append(best_match)
            if len(self.sign_history) > self.history_size:
                self.sign_history.pop(0)
            
            # Check if we have consistent detection
            if len(self.sign_history) >= self.history_size:
                most_common = max(set(self.sign_history), key=self.sign_history.count)
                consistent_count = self.sign_history.count(most_common)
                
                # Only proceed if sign is consistent and buffer time has passed
                if consistent_count >= self.history_size * 0.8:  # 80% consistency
                    elapsed_time = time.time() - self.detection_start_time
                    if elapsed_time >= self.buffer_time:
                        self.last_sign_time = time.time()
                        self.sign_history.clear()  # Reset history after accepting sign
                        self.detection_start_time = None  # Reset detection timer
                        return most_common
        else:
            self.detection_start_time = None  # Reset detection timer if no match
            
        return None

    def draw_hand_box(self, frame, hand_landmarks, hand_type=""):
        """Draw a bounding box around the hand landmarks with sign text."""
        frame_height, frame_width = frame.shape[:2]
        points = np.array([[lm.x * frame_width, lm.y * frame_height] for lm in hand_landmarks.landmark])
        
        padding = 20
        x_min, x_max = int(min(points[:, 0])) - padding, int(max(points[:, 0])) + padding
        y_min, y_max = int(min(points[:, 1])) - padding, int(max(points[:, 1])) + padding
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame_width, x_max)
        y_max = min(frame_height, y_max)
        
        # Draw box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # Draw center point
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)

        # Draw current sign next to the hand box
        if self.current_sign:
            sign_text = f"{hand_type}: {self.current_sign}"
            cv2.putText(frame, sign_text, (x_max + 10, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw loading bar background
            bar_width = 150
            bar_height = 10
            bar_x = x_max + 10
            bar_y = center_y + 20
            cv2.rectangle(frame, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (70, 70, 70), -1)
            
            # Calculate and draw loading progress
            if self.detection_start_time is not None:
                elapsed_time = time.time() - self.detection_start_time
                progress = min(1.0, elapsed_time / self.buffer_time)
                
                # Draw filled portion of loading bar
                filled_width = int(bar_width * progress)
                cv2.rectangle(frame,
                            (bar_x, bar_y),
                            (bar_x + filled_width, bar_y + bar_height),
                            (0, 255, 255), -1)
                
                # Draw progress percentage
                cv2.putText(frame, f"{int(progress * 100)}%",
                           (bar_x + bar_width + 5, bar_y + bar_height),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
        elif detected_sign == "-" and not self.tts.is_busy():  # Speak
            sentence_text = "".join(self.sentence).strip()
            if sentence_text:
                # Apply autocorrect before speaking
                corrected_text = autocorrect(sentence_text)
                self.tts.speak(corrected_text)
                self.sentence.clear()
        else:
            self.sentence.append(detected_sign)

    def draw_ui(self, frame):
        """Draw the UI panel with current sentence."""
        # Create semi-transparent overlay for the bottom panel
        panel_height = 100  # Increased height for two lines
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - panel_height), 
                     (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Display current sentence and its autocorrected version
        sentence_text = "".join(self.sentence)
        corrected_text = autocorrect(sentence_text) if sentence_text.strip() else ""
        
        cv2.putText(frame, "Current: " + sentence_text,
                   (20, frame.shape[0] - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if corrected_text and corrected_text != sentence_text:
            cv2.putText(frame, "Corrected: " + corrected_text,
                       (20, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display controls
        controls = "Controls: '_' for space, '>' for backspace, '-' for speak"
        cv2.putText(frame, controls, (20, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Show speaking status
        if self.tts.is_busy():
            cv2.putText(frame, "Speaking...", (frame.shape[1]-150, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def run(self):
        """Main loop for sign detection and sentence building."""
        cap = cv2.VideoCapture(0)
        
        # Get the screen resolution
        screen_width = 1280
        screen_height = 720
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

        print("\nSign Language Recognition Started")
        print("\nLoaded signs:", ', '.join(sorted(self.signs_data.keys())))
        print("\nControls:")
        print("'_' - Space")
        print("'>' - Backspace")
        print("'-' - Speak")
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
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get hand type
                    hand_type = results.multi_handedness[idx].classification[0].label
                    
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )
                    # Draw box with sign text
                    self.draw_hand_box(frame, hand_landmarks, hand_type)

            # Detect and process signs
            detected_sign = self.detect_sign(results)
            if detected_sign:
                self.update_sentence(detected_sign)

            # Draw UI
            self.draw_ui(frame)
            
            # Show the frame
            cv2.imshow('Sign Language Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = SignTester()
    tester.run()

