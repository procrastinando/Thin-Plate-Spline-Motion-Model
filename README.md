# [CVPR2022] Thin-Plate Spline Motion Model for Image Animation

This is a fork of the original repository:

[**Thin-Plate-Spline-Motion-Model**]([https://arxiv.org/abs/2203.14367](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)) **

There is a WebUI, as well as an automatic crop of images and videos to a square shape.

### 1.1 Automatic Installation
First install Anaconda ang git. Then clone the repository:
```
git clone https://github.com/procrastinando/Thin-Plate-Spline-Motion-Model
cd Thin-Plate-Spline-Motion-Model
```
Run `install-windows.bat` for windows or `install-linux.sh` for a debian based OS.

### 1.2. Manual Installation
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
### 2. Download the pretrained models and put them in `checkpoints` directory:
```
https://disk.yandex.com/d/i08z-kCuDGLuYA
https://disk.yandex.com/d/vk5dirE6KNvEXQ
https://disk.yandex.com/d/IVtro0k2MVHSvQ
https://disk.yandex.com/d/B3ipFzpmkB1HIA
```

### 3. Run the WebUI
Run `start-windows.bat` for windows or `start-linux.sh` for a debian based OS.

## 4. Use colab
4.1. Install requirements:
```
!https://github.com/procrastinando/Thin-Plate-Spline-Motion-Model.git
%cd Thin-Plate-Spline-Motion-Model
!pip install -r requirements.txt
```
4.2. Download pretrained models (only vox in this case):
```
!mkdir checkpoints
!pip3 install wldhx.yadisk-direct
!curl -L $(yadisk-direct https://disk.yandex.com/d/i08z-kCuDGLuYA) -o checkpoints/vox.pth.tar
# !curl -L $(yadisk-direct https://disk.yandex.com/d/vk5dirE6KNvEXQ) -o checkpoints/taichi.pth.tar
# !curl -L $(yadisk-direct https://disk.yandex.com/d/IVtro0k2MVHSvQ) -o checkpoints/mgif.pth.tar
# !curl -L $(yadisk-direct https://disk.yandex.com/d/B3ipFzpmkB1HIA) -o checkpoints/ted.pth.tar
```
4.3. Run the WebUI
```
!python app.py
```
![image](https://github.com/procrastinando/Thin-Plate-Spline-Motion-Model/assets/74340724/759031b6-5b68-43cb-b142-a1664c1a63d4)
