#!/bin/bash

# This script downloads the required pre-trained models from Google Drive.

# Create the 'models' directory if it doesn't already exist.
echo "Creating 'models' directory..."
mkdir -p models

echo "Downloading models..."

# Download Ball Detection Model
echo "--> Downloading football-ball-detection.pt..."
gdown -O "models/football-ball-detection.pt" "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"

# Download Player Detection Model
echo "--> Downloading football-player-detection.pt..."
gdown -O "models/football-player-detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"

# Download Pitch Keypoint Detection Model
echo "--> Downloading football-pitch-detection.pt..."
gdown -O "models/football-pitch-detection.pt" "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

echo ""
echo "✅ All models downloaded successfully into the 'models' folder!"