#!/bin/bash

# Downloads the required models from HuggingFace (martinjolif).
# YOLO detectors + MobileNetV3 jersey classifier used for team assignment.

mkdir -p models

echo "Downloading models from HuggingFace..."

echo "--> yolo-football-player-detection.pt..."
curl -sL --fail -o "models/football-player-detection.pt" \
  "https://huggingface.co/martinjolif/yolo-football-player-detection/resolve/main/yolo-football-player-detection.pt"

echo "--> yolo-football-ball-detection.pt..."
curl -sL --fail -o "models/football-ball-detection.pt" \
  "https://huggingface.co/martinjolif/yolo-football-ball-detection/resolve/main/yolo-football-ball-detection.pt"

echo "--> yolo-football-pitch-detection.pt..."
curl -sL --fail -o "models/football-pitch-detection.pt" \
  "https://huggingface.co/martinjolif/yolo-football-pitch-detection/resolve/main/yolo-football-pitch-detection.pt"

echo "--> mobilenetv3-football-jersey-classification.pth..."
curl -sL --fail -o "models/football-jersey-classification.pth" \
  "https://huggingface.co/martinjolif/mobilenetv3-football-jersey-classification/resolve/main/mobilenetv3-football-jersey-classification.pth"

echo ""
echo "Done. Expected files in models/:"
ls -la models/*.pt models/*.pth
