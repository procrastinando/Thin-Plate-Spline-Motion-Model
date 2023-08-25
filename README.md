# [CVPR2022] Thin-Plate Spline Motion Model for Image Animation

This is a fork of the original repository:

[**BreadcrumbsThin-Plate-Spline-Motion-Model**]([https://arxiv.org/abs/2203.14367](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)) **

There is a WebUI, as well as an automatic crop of images and videos to a square shape.

### Automatic Installation
First install Anaconda.
Download the repository and run `install-windows.bat` for windows or `install-linux.sh` for a debian based OS.

### Manual Installation
First install Anaconda, Then run the commands:
```
git clone https://github.com/procrastinando/Thin-Plate-Spline-Motion-Model
cd Thin-Plate-Spline-Motion-Model
conda create --name image-animation python=3.9.13 -y
conda activate image-animation
```
If you have CUDA, run:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If you dont:
```
pip install torch torchvision torchaudio
```
Install requirements
```
pip install -r requirements.txt
```

### Run the WebUI
Run `start-windows.bat` for windows or `start-linux.sh` for a debian based OS.
