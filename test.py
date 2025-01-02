import cv2
import mediapipe as mp
import numpy as np
import json
import time
import autocorrect
import requests
import pygame
import os

def text_to_speech(sentence):
    
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
        
        # Make the POST request to the API
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            # Save the audio file
            with open("result.mp3", "wb") as file:
                file.write(response.content)
            pygame.mixer.init()
            pygame.mixer.music.load("result.mp3")
            pygame.mixer.music.play()
        
            # Wait until playback is done
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            pygame.mixer.music.unload()
            if os.path.exists("result.mp3"):
                os.remove("result.mp3")
        else:
            print(f"Error: Unable to generate speech. Status code: {response.status_code} Error: {response.text}")



class SignTester:
    def __init__(self):
        self.corrected_sentence = ""

        # Initialize MediaPipe
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
            model_complexity=0
        )
        
        self.trained_signs = self.load_sign_data()
        self.sentence = []
        self.last_sign_time = time.time()
        self.buffer_time = 3.0  # Buffer time between sign detections
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
        if not current_landmarks['left_hand'] and not current_landmarks['right_hand']:
            return None  # Do nothing if no hand landmarks are detected
    
        best_match = None
        best_score = 0.7  # Minimum threshold
        
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
    
    def realtime_sign(self, results):
        """Detect the sign from current landmarks and return the best match."""
        current_landmarks = self.extract_landmarks(results)

        if not current_landmarks['left_hand'] and not current_landmarks['right_hand']:
            return None  # Do nothing if no hand landmarks are detected
    
        best_match = None
        best_score = 0.6  # Minimum threshold
        
        for sign_name, sign_data in self.trained_signs.items():
            score = self.compare_landmarks(current_landmarks, sign_data)
            if score > best_score:
                best_score = score
                best_match = sign_name
        
        # Use buffer time to prevent rapid detection
        if best_match:
            return best_match
        return None
   
    def update_sentence(self, detected_sign):
        if detected_sign == "n/a":
            return  # Do nothing if the detected sign is "n/a"
    
        if detected_sign == "_":
            self.sentence.append(" ")
        elif detected_sign == ">":
            if self.sentence:
                self.sentence.pop()
        elif detected_sign == "-":  # Enter key to convert to speech
            # Convert the current sentence to a string
            sentence_text = "".join(self.sentence)
            # Apply autocorrection
            self.corrected_sentence = autocorrect.autocorrect(sentence_text)
            
            # Call the text-to-speech function
            text_to_speech(self.corrected_sentence)
            
            # Reset the sentence after playing the audio
            self.sentence.clear()
            self.corrected_sentence = ""
        else:
            self.sentence.append(detected_sign)
            # Update the autocorrected sentence
            sentence_text = "".join(self.sentence)
            self.corrected_sentence = autocorrect.autocorrect(sentence_text)

    def draw_panel(self, display):
    
        cv2.rectangle(display, (20, 20), (620, 700), (50, 50, 50), -1)

        # Display the constructed sentence (before autocorrect)
        raw_sentence_text = "".join(self.sentence)
        words = raw_sentence_text.split()
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

        y_pos = 60
        for line in lines:
            cv2.putText(display, line, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_pos += 40

        # Display the autocorrected sentence
        cv2.putText(display, "Autocorrected:", (40, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, self.corrected_sentence, (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the last detected sign
        
        cv2.putText(display, f"Sign: {self.sign}",
                        (40, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show buffer progress bar
        time_since_last = time.time() - self.last_sign_time
        if time_since_last < self.buffer_time:
            progress = int((time_since_last / self.buffer_time) * 100)
            cv2.rectangle(display, (40, 650), (40 + progress * 5, 670), (255, 0, 0), -1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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

            self.sign = self.realtime_sign(results)
            # Draw the panel with the text
            self.draw_panel(display)

            if results.face_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

            cv2.imshow('Training', frame)
            
            cv2.imshow('Sign Language to Text', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = SignTester()
    tester.run()

