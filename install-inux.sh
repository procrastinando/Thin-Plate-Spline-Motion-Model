#!/bin/bash

git clone https://github.com/procrastinando/Thin-Plate-Spline-Motion-Model.git
cd Thin-Plate-Spline-Motion-Model
conda create --name image-animation python=3.9.13 -y
conda activate image-animation
# Check if CUDA is installed
if command -v nvcc > /dev/null; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip3 install torch torchvision torchaudio
fi

# Install requirements
pip install -r requirements.txt
python app.py