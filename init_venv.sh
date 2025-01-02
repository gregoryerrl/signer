#!/bin/bash

# Define the virtual environment directory
VENV_DIR="venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Activating virtual environment..."
    source $VENV_DIR/bin/activate
else
    echo "Virtual environment activation script not found."
    exit 1
fi

# Install required packages
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip --break-system-packages
    pip install -r requirements.txt --break-system-packages
    echo "Dependencies installed."
else
    echo "requirements.txt not found. Please create it with your dependencies listed."
fi

# Notify the user
echo "Setup complete. To activate the virtual environment, run:"
echo "source $VENV_DIR/bin/activate"
