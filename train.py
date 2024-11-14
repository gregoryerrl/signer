import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
import sys

class ASLRecorder:
    def __init__(self, save_dir="recorded_signs", headless=False):
        # Initialize MediaPipe components
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize models
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Ensure save directory exists
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.current_sign = None
        self.recorded_frames = []
        self.headless = headless
        
        # Load existing data if available
        self.landmarks_data = self._load_existing_data()
    
    def _load_existing_data(self):
        json_path = os.path.join(self.save_dir, "landmarks_data.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_landmarks(self):
        """Save recorded landmarks to JSON file"""
        if not self.recorded_frames or not self.current_sign:
            return
        
        # Average the landmarks across recorded frames
        avg_landmarks = {
            'face': np.mean([frame['face'] for frame in self.recorded_frames if frame['face'] is not None], axis=0).tolist() if any(frame['face'] is not None for frame in self.recorded_frames) else None,
            'left_hand': np.mean([frame['left_hand'] for frame in self.recorded_frames if frame['left_hand'] is not None], axis=0).tolist() if any(frame['left_hand'] is not None for frame in self.recorded_frames) else None,
            'right_hand': np.mean([frame['right_hand'] for frame in self.recorded_frames if frame['right_hand'] is not None], axis=0).tolist() if any(frame['right_hand'] is not None for frame in self.recorded_frames) else None
        }
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sign_key = f"{self.current_sign}_{timestamp}"
        
        # Update landmarks data
        self.landmarks_data[sign_key] = avg_landmarks
        
        # Save to file
        json_path = os.path.join(self.save_dir, "landmarks_data.json")
        with open(json_path, 'w') as f:
            json.dump(self.landmarks_data, f, indent=2)
        
        print(f"Saved landmarks for sign: {self.current_sign}")
    
    def _extract_landmarks(self, results_holistic, results_hands):
        """Extract landmarks from MediaPipe results"""
        # Extract face landmarks
        face_landmarks = None
        if results_holistic.face_landmarks:
            face_landmarks = [[lm.x, lm.y, lm.z] for lm in results_holistic.face_landmarks.landmark]
        
        # Extract hand landmarks
        left_hand = None
        right_hand = None
        
        if results_hands.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                hand_type = results_hands.multi_handedness[hand_idx].classification[0].label
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                
                if hand_type == "Left":
                    left_hand = landmarks
                else:
                    right_hand = landmarks
        
        return {
            'face': face_landmarks,
            'left_hand': left_hand,
            'right_hand': right_hand
        }
    
    def process_frame(self, frame):
        """Process a single frame and return the processed frame and landmarks"""
        if frame is None:
            return None, None
        
        # Flip frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results_holistic = self.holistic.process(rgb_frame)
        results_hands = self.hands.process(rgb_frame)
        
        # Extract landmarks
        landmarks = self._extract_landmarks(results_holistic, results_hands)
        
        if not self.headless:
            # Draw landmarks
            if results_holistic.face_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    results_holistic.face_landmarks,
                    self.mp_holistic.FACEMESH_CONTOURS,
                    self.mp_draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    self.mp_draw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
            
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                    )
        
        # Record landmarks if recording is active
        if self.is_recording:
            self.recorded_frames.append(landmarks)
        
        return frame, landmarks
    
    def start_recording(self, sign_name):
        """Start recording a new sign"""
        self.is_recording = True
        self.current_sign = sign_name
        self.recorded_frames = []
        print(f"Started recording sign: {sign_name}")
    
    def stop_recording(self):
        """Stop recording and save the landmarks"""
        if self.is_recording:
            self._save_landmarks()
            self.is_recording = False
            self.current_sign = None
            self.recorded_frames = []
            print("Recording stopped")

def main():
    # Try to determine if we can use GUI
    has_gui = True
    try:
        test_window = cv2.namedWindow("test")
        cv2.destroyWindow("test")
    except:
        has_gui = False
        print("Running in headless mode (no GUI available)")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    recorder = ASLRecorder(headless=not has_gui)
    
    print("\nCommands:")
    print("'r' - Start recording (enter sign name when prompted)")
    print("'s' - Stop recording")
    print("'q' - Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        processed_frame, landmarks = recorder.process_frame(frame)
        
        if has_gui:
            # Display frame
            cv2.imshow('ASL Sign Recorder', processed_frame)
            key = cv2.waitKey(1) & 0xFF
        else:
            # In headless mode, use input() for commands
            key = ord(input("Enter command (r/s/q): ").lower() or 'x')
        
        if key in [ord('q'), 27]:  # q or ESC
            break
        elif key == ord('r') and not recorder.is_recording:
            sign_name = input("Enter the name of the sign to record: ")
            recorder.start_recording(sign_name)
        elif key == ord('s') and recorder.is_recording:
            recorder.stop_recording()
    
    # Clean up
    cap.release()
    if has_gui:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)