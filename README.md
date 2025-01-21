# Sign Language Recognition System

A Python-based sign language training and recognition system using MediaPipe Hands for gesture detection.

## Updating Your Repository

1. Pull the latest changes:

```bash
git pull origin main
```

2. Activate your virtual environment:

```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Update dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training New Signs (`train.py`)

The training program allows you to record and save new hand gestures/signs.

1. Run the training program:

```bash
python train.py
```

2. Controls:

- `R` - Start/Stop Recording (records 30 frames)
- `L` - List all saved signs
- `P` - Toggle preview mode
- `Q` - Quit

3. Training Process:

   - Press `R` to start recording
   - Hold your hand steady in the desired position
   - Wait for the red progress bar to complete
   - Enter a name for the sign when prompted
   - Press Enter without a name to discard the recording

4. Special Characters:
   - Space: Train a sign and name it "\_" (underscore)
   - Backspace: Train a sign and name it ">" (greater than)
   - Speak/Enter: Train a sign and name it "-" (hyphen)
     These special characters are used by the system for text manipulation and speech output.

Tips:

- Keep your hand within the camera frame
- Hold the sign steady during recording
- The bounding box turns red during recording
- Signs are automatically saved with backups in `sign_data/`

### File Structure

```
signer/
├── train.py           # Training program
├── requirements.txt   # Package dependencies
├── sign_data/        # Directory for saved signs
│   ├── signs.json    # Main signs database
│   └── backups/      # Automatic backups
```

### Data Management

- Signs are saved in `sign_data/signs.json`
- Automatic backups are created in `sign_data/backups/`
- Each backup includes a timestamp

## Requirements

- Python 3.8 or higher
- Webcam
- Packages listed in requirements.txt:
  - mediapipe>=0.10.8
  - opencv-python>=4.8.0
  - numpy>=1.24.0
  - pygame>=2.5.0

## Changes in This Update

- Removed MediaPipe Holistic in favor of MediaPipe Hands for better performance
- Added automatic backup system for sign data
- Improved training interface with:
  - Progress bar for recording
  - Preview mode
  - Color-coded bounding boxes
  - Better error handling
- Simplified dependencies

## Troubleshooting

1. If the camera doesn't open:

   - Check if another program is using the camera
   - Try restarting your computer
   - Verify camera permissions

2. If hand detection is unstable:

   - Ensure good lighting
   - Keep your hand within frame
   - Try adjusting your distance from the camera

3. If signs aren't saving:
   - Check write permissions in the sign_data directory
   - Verify that the JSON file isn't open in another program
