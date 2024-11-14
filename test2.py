import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

class DynamicASLRecognizer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Motion tracking
        self.trajectory_length = 20  # Number of points to track
        self.index_trajectory = deque(maxlen=self.trajectory_length)
        
        # Define motion patterns for dynamic signs
        self.motion_patterns = {
            'J': {
                'description': 'hook motion downward',
                'pattern': [(0, 1), (0.1, 1), (0.2, 0.9), (0.2, 0.8), (0.1, 0.7)]
            },
            'Z': {
                'description': 'zig-zag motion',
                'pattern': [(0, 0), (1, 0), (0, 1), (1, 1)]
            }
        }
        
        # State variables
        self.is_recording = False
        self.current_motion = []
        self.detected_sign = None
        self.confidence = 0.0
        
    def normalize_trajectory(self, trajectory):
        """Normalize trajectory points to 0-1 range"""
        if not trajectory:
            return []
            
        points = np.array(trajectory)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        
        # Avoid division by zero
        x_range = max_x - min_x or 1
        y_range = max_y - min_y or 1
        
        # Fixed: Changed 'mp' to 'point' and corrected the list comprehension
        normalized = [((point[0] - min_x) / x_range, (point[1] - min_y) / y_range) 
                     for point in points]
        return normalized
    
    def calculate_dtw(self, seq1, seq2):
        """Calculate Dynamic Time Warping distance between two sequences"""
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.zeros((n + 1, m + 1))
        dtw_matrix.fill(float('inf'))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(np.array(seq1[i-1]) - np.array(seq2[j-1]))
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                            dtw_matrix[i, j-1],
                                            dtw_matrix[i-1, j-1])
        
        return dtw_matrix[n, m]
    
    def match_pattern(self, trajectory):
        """Match the current trajectory against known patterns"""
        if len(trajectory) < 4:  # Minimum points needed
            return None, 0.0
            
        normalized = self.normalize_trajectory(trajectory)
        
        # Added error checking for empty normalized trajectory
        if not normalized:
            return None, 0.0
            
        best_match = None
        best_score = float('inf')
        
        for sign, pattern_info in self.motion_patterns.items():
            pattern = pattern_info['pattern']
            try:
                distance = self.calculate_dtw(normalized, pattern)
                
                if distance < best_score:
                    best_score = distance
                    best_match = sign
            except Exception as e:
                print(f"Error matching pattern for {sign}: {e}")
                continue
        
        # Convert distance to confidence (inverse relationship)
        confidence = max(0, 1 - (best_score / 10))  # Adjust denominator as needed
        return best_match, confidence
    
    def process_frame(self, frame):
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Track index finger tip
                index_tip = hand_landmarks.landmark[8]
                x, y = int(index_tip.x * width), int(index_tip.y * height)
                
                # Add point to trajectory
                self.index_trajectory.append((x, y))
                
                # Draw trajectory
                points = list(self.index_trajectory)
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
        
        # Match pattern if recording
        if self.is_recording and len(self.index_trajectory) >= 4:
            self.detected_sign, self.confidence = self.match_pattern(list(self.index_trajectory))
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    recognizer = DynamicASLRecognizer()
    
    print("Starting Dynamic ASL recognition...")
    print("Press 'r' to start/stop recording motion")
    print("Press 'c' to clear trajectory")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
            
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Process frame
        frame = recognizer.process_frame(frame)
        
        # Draw status and results
        status = "Recording" if recognizer.is_recording else "Not Recording"
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if recognizer.detected_sign:
            cv2.putText(frame, f"Sign: {recognizer.detected_sign}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {recognizer.confidence:.2f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instruction box
        cv2.rectangle(frame, (10, 400), (400, 460), (0, 0, 0), -1)
        cv2.putText(frame, "R: Record  C: Clear  Q: Quit", (20, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Dynamic ASL Recognition', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.is_recording = not recognizer.is_recording
            if recognizer.is_recording:
                recognizer.index_trajectory.clear()
                recognizer.detected_sign = None
        elif key == ord('c'):
            recognizer.index_trajectory.clear()
            recognizer.detected_sign = None
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()