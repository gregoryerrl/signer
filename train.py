import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime

class SignTrainer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        
        # File paths
        self.data_dir = "sign_data"
        self.data_file = os.path.join(self.data_dir, "signs.json")
        self.backup_dir = os.path.join(self.data_dir, "backups")
        
        # Load existing data
        self.signs_data = self.load_sign_data()
        
        # Recording settings
        self.recording_frames = []
        self.is_recording = False
        self.is_countdown = False
        self.countdown_start = 0
        self.frames_to_record = 30  # Record 30 frames for better accuracy
        self.current_sign = None
        self.show_preview = True
        
        # Input box settings
        self.input_text = ""
        self.is_inputting = False
        self.input_cursor_visible = True
        self.last_cursor_toggle = time.time()
        self.cursor_blink_rate = 0.5  # seconds

    def load_sign_data(self):
        """Load existing sign data from JSON file."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {}

    def save_sign_data(self):
        """Save the collected sign data to JSON with backup."""
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

        # Create backup with timestamp
        backup_file = os.path.join(self.backup_dir, f"signs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                with open(backup_file, 'w') as bf:
                    bf.write(f.read())

        # Save current data
        with open(self.data_file, 'w') as f:
            json.dump(self.signs_data, f, indent=4)
        print(f"\nData saved to {self.data_file}")
        print(f"Backup created at {backup_file}")

    def capture_landmarks(self, results):
        """Extract and normalize landmarks from MediaPipe results."""
        landmarks = {'left_hand': None, 'right_hand': None}

        if not results.multi_hand_landmarks:
            return landmarks

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_type = results.multi_handedness[hand_idx].classification[0].label.lower()
            landmarks[f"{hand_type}_hand"] = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

        return landmarks

    def add_sign(self, sign_name, landmarks_sequence):
        """Add a new sign to the dataset using multiple frames."""
        # Average the landmarks across frames for more stable detection
        avg_landmarks = {'left_hand': None, 'right_hand': None}
        valid_frames = [frame for frame in landmarks_sequence if frame['left_hand'] or frame['right_hand']]
        
        if not valid_frames:
            return False

        for hand in ['left_hand', 'right_hand']:
            hand_frames = [frame[hand] for frame in valid_frames if frame[hand]]
            if hand_frames:
                avg_landmarks[hand] = np.mean(hand_frames, axis=0).tolist()

        self.signs_data[sign_name] = avg_landmarks
        print(f"\nAdded sign: {sign_name} (averaged from {len(valid_frames)} frames)")
        return True

    def draw_hand_box(self, frame, hand_landmarks):
        """Draw a bounding box around the hand landmarks."""
        frame_height, frame_width = frame.shape[:2]
        points = np.array([[lm.x * frame_width, lm.y * frame_height] for lm in hand_landmarks.landmark])
        
        padding = 20
        x_min, x_max = int(min(points[:, 0])) - padding, int(max(points[:, 0])) + padding
        y_min, y_max = int(min(points[:, 1])) - padding, int(max(points[:, 1])) + padding
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame_width, x_max)
        y_max = min(frame_height, y_max)
        
        # Draw box with different colors based on recording status
        color = (0, 0, 255) if self.is_recording else (0, 255, 255)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw center point
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        cv2.circle(frame, (center_x, center_y), 4, color, -1)

    def draw_input_box(self, frame):
        """Draw an input box for sign name."""
        # Create semi-transparent overlay for input box
        overlay = frame.copy()
        box_width = 400
        box_height = 100
        start_x = (frame.shape[1] - box_width) // 2
        start_y = (frame.shape[0] - box_height) // 2
        
        # Draw background box
        cv2.rectangle(overlay, 
                     (start_x, start_y),
                     (start_x + box_width, start_y + box_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame,
                     (start_x, start_y),
                     (start_x + box_width, start_y + box_height),
                     (255, 255, 255), 2)
        
        # Draw title
        cv2.putText(frame, "Enter sign name:",
                   (start_x + 10, start_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw input text with blinking cursor
        text_x = start_x + 10
        text_y = start_y + 70
        
        # Toggle cursor visibility
        if time.time() - self.last_cursor_toggle > self.cursor_blink_rate:
            self.input_cursor_visible = not self.input_cursor_visible
            self.last_cursor_toggle = time.time()
        
        # Draw text with cursor
        display_text = self.input_text
        if self.input_cursor_visible:
            display_text += "|"
            
        cv2.putText(frame, display_text,
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def handle_key_input(self, key):
        """Handle keyboard input for the input box."""
        if not self.is_inputting:
            return False
            
        if key == 27:  # ESC
            self.is_inputting = False
            self.input_text = ""
            return False
        elif key == 13 or key == ord('\r'):  # Enter
            if self.input_text.strip():
                self.is_inputting = False
                return True
            return False
        elif key == 8 or key == 127 or key == ord('\b'):  # Backspace (multiple platform support)
            if self.input_text:  # Only remove if there's text
                self.input_text = self.input_text[:-1]
        elif 32 <= key <= 126:  # Printable characters
            self.input_text += chr(key)
        return False

    def draw_ui(self, frame):
        """Draw UI elements on the frame."""
        # Create semi-transparent overlay for instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Display instructions
        cv2.putText(frame, "Controls:", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "R - Start/Stop Recording", (30, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "L - List saved signs", (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "P - Toggle preview", (30, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Q - Quit", (30, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show countdown if active
        if self.is_countdown:
            elapsed = time.time() - self.countdown_start
            remaining = max(0, 5 - int(elapsed))
            if remaining > 0:
                # Draw large countdown number in center
                font_scale = 5.0
                thickness = 5
                text = str(remaining)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
                cv2.putText(frame, "Get ready!", (text_x - 50, text_y - 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                self.is_countdown = False
                self.is_recording = True
                print("\nRecording started!")

        # Show recording status and progress
        if self.is_recording:
            cv2.putText(frame, "Recording...", (frame.shape[1]-200, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            progress = len(self.recording_frames) / self.frames_to_record
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (frame.shape[1]-200, 50), 
                        (frame.shape[1]-200+bar_width, 70), (0, 0, 255), -1)

        # Show current sign if in preview mode
        if self.show_preview and self.current_sign:
            cv2.putText(frame, f"Current Sign: {self.current_sign}", (20, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    def run(self):
        """Main training loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\nSign Language Trainer Started")
        print("\nControls:")
        print("'R' - Start/Stop Recording")
        print("'L' - List saved signs")
        print("'P' - Toggle preview mode")
        print("'Q' - Quit")
        print("\nTip: Hold your hand steady while recording")

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
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )
                    self.draw_hand_box(frame, hand_landmarks)

            # Handle recording
            if self.is_recording and not self.is_countdown:
                landmarks = self.capture_landmarks(results)
                if landmarks['left_hand'] or landmarks['right_hand']:
                    self.recording_frames.append(landmarks)
                
                if len(self.recording_frames) >= self.frames_to_record:
                    self.is_recording = False
                    self.is_inputting = True
                    self.input_text = ""

            # Draw UI elements
            self.draw_ui(frame)
            
            # Draw input box if active
            if self.is_inputting:
                self.draw_input_box(frame)

            cv2.imshow('Sign Training', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if self.is_inputting:
                if self.handle_key_input(key):
                    sign_name = self.input_text.strip().lower()
                    if self.add_sign(sign_name, self.recording_frames):
                        self.save_sign_data()
                    else:
                        print("Failed to add sign: No valid hand landmarks detected")
                    self.recording_frames = []
                    self.input_text = ""
            else:
                if key == ord('r'):
                    if not self.is_recording and not self.is_countdown:
                        print("\nStarting countdown...")
                        self.is_countdown = True
                        self.countdown_start = time.time()
                        self.recording_frames = []
                elif key == ord('l'):
                    signs = sorted(self.signs_data.keys())
                    if signs:
                        print("\nSaved signs:", ', '.join(signs))
                    else:
                        print("\nNo signs saved yet")
                elif key == ord('p'):
                    self.show_preview = not self.show_preview
                    print(f"\nPreview mode {'enabled' if self.show_preview else 'disabled'}")
                elif key == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    trainer = SignTrainer()
    trainer.run()
