@echo off

conda create --name image-animation python=3.9.13 -y
call conda activate image-animation
:: Install PyTorch
if defined CUDA_PATH (
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    pip3 install torch torchvision torchaudio
)

:: Install requirements
pip install -r requirements.txt
python app.py
pause

