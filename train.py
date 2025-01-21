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
        self.frames_to_record = 30  # Record 30 frames for better accuracy
        self.current_sign = None
        self.show_preview = True

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

    def draw_ui(self, frame):
        """Draw UI elements on the frame."""
        # Create a semi-transparent overlay for instructions
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
            if self.is_recording:
                landmarks = self.capture_landmarks(results)
                if landmarks['left_hand'] or landmarks['right_hand']:
                    self.recording_frames.append(landmarks)
                
                if len(self.recording_frames) >= self.frames_to_record:
                    self.is_recording = False
                    sign_name = input("\nEnter sign name (or press Enter to discard): ").strip().lower()
                    if sign_name:
                        if self.add_sign(sign_name, self.recording_frames):
                            self.save_sign_data()
                        else:
                            print("Failed to add sign: No valid hand landmarks detected")
                    self.recording_frames = []

            # Draw UI
            self.draw_ui(frame)
            cv2.imshow('Sign Training', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                if not self.is_recording:
                    print("\nStarting recording...")
                    self.is_recording = True
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
