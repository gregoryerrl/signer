# Sign Language Recognition System

A real-time sign language recognition system that allows you to train custom signs and translate them into text and speech.

## Features

- Real-time hand tracking and sign detection
- Custom sign training with visual feedback
- Text-to-speech output
- Autocorrect functionality for natural sentences
- Support for special characters and commands
- Automatic data backup system

## Requirements

- Python 3.8 or higher
- Webcam
- macOS, Windows, or Linux

## Setup Instructions

1. **Update your codes**

   ```bash
   git pull
   ```

2. **Create and Activate Virtual Environment**

   - For Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - For macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training New Signs (`train.py`)

1. Run the training program:

   ```bash
   python train.py
   ```

2. Controls:

   - Press 'R' to start recording a sign
   - Wait for the 5-second countdown
   - Hold the sign steady during recording
   - Type the sign name when prompted
   - Press Enter to save or ESC to cancel
   - Press 'L' to list saved signs
   - Press 'Q' to quit

3. Training Tips:
   - Keep your hand steady during recording
   - Use good lighting
   - Train in different positions for better recognition
   - Record multiple samples for better accuracy

### Testing Signs (`test.py`)

1. Run the testing program:

   ```bash
   python test.py
   ```

2. Controls:

   - Show a sign and hold steady for 2 seconds
   - Wait for the loading bar to fill completely
   - Use special signs:
     - "\_" (underscore) for space
     - ">" (greater than) for backspace
     - "-" (hyphen) to speak the sentence
   - Press 'Q' to quit

3. Features:
   - Real-time sign detection
   - Visual feedback with loading bar
   - Autocorrect for natural sentences
   - Text-to-speech output

## Special Signs

Train these signs for special functions:

- Space: Train a sign and name it "\_" (underscore)
- Backspace: Train a sign and name it ">" (greater than)
- Speak/Enter: Train a sign and name it "-" (hyphen)

## Data Management

- Signs are saved in `sign_data/signs.json`
- Automatic backups are created in `sign_data/backups/`
- Each backup includes timestamp for version control

## Troubleshooting

1. **Camera Issues**

   - Check if another application is using the camera
   - Ensure proper lighting conditions
   - Verify camera permissions

2. **Recognition Issues**

   - Retrain signs with different positions
   - Ensure consistent lighting
   - Hold signs steady during detection
   - Check the confidence threshold in settings

3. **Audio Issues**
   - Verify system audio is working
   - Check if text-to-speech is enabled on your system
   - Restart the program if speech stops working

## Updates and Maintenance

1. **Updating the Repository**

   ```bash
   git pull origin main
   ```

2. **Updating Dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

## System Requirements

- Minimum 4GB RAM
- Webcam with minimum 720p resolution
- Processor: Intel i3/AMD equivalent or better
- Free disk space: 500MB
- Internet connection (for initial setup only)
