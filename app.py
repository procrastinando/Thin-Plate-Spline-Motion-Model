import imageio
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import os
import gradio as gr
import torch
from PIL import Image
import cv2

def crop_video_to_square(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = min(width, height)
    x = (width - size) // 2
    y = (height - size) // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('square_' + video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (size, size))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cropped_frame = frame[y:y+size, x:x+size]
            out.write(cropped_frame)
        else:
            break

    cap.release()
    out.release()

def crop_image_to_square(image_path):
    with Image.open(image_path) as im:
        width, height = im.size
        if width == height:
            return im
        elif width > height:
            left = (width - height) // 2
            right = left + height
            top = 0
            bottom = height
        else:
            top = (height - width) // 2
            bottom = top + width
            left = 0
            right = width

        square_image = im.crop((left, top, right, bottom))
        directory_path = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        square_image.save(os.path.join(directory_path, f'square_{filename}'))

    return square_image

def image_animation(dataset_name, source_image_path, driving_video_path):
    image = Image.fromarray(source_image_path)
    image.save('assets/input.jpg')
    source_image_path = 'assets/input.jpg'

    device = torch.device('cuda:0')
    output_video_path = './generated.mp4'
    config_path = 'config/vox-256.yaml'
    checkpoint_path = 'checkpoints/vox.pth.tar'
    predict_mode = 'relative' # ['standard', 'relative', 'avd']
    find_best_frame = False # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

    pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
    if(dataset_name == 'ted'): # for ted, the resolution is 384*384
        pixel = 384

    source_image = imageio.imread(source_image_path)
    reader = imageio.get_reader(driving_video_path)

    source_image = resize(source_image, (pixel, pixel))[..., :3]

    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]

    from demo import load_checkpoints
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)

    from demo import make_animation
    from skimage import img_as_ubyte

    if predict_mode=='relative' and find_best_frame:
        from demo import find_best_frame as _find
        #i = _find(source_image, driving_video, device.type=='cuda')
        i = _find(source_image, driving_video, device.type=='cpu')
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
        predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)

    #save resulting video
    imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

    return output_video_path

dataset = gr.Dropdown(['vox', 'taichi', 'ted', 'mgif'], label="Dataset")
image = gr.Image(label="Input Image")
video = gr.Video(label="Input Video")
video_out = gr.Video(label="Output Video")

# webapp with gradio
gr.Interface(fn=image_animation, inputs=[dataset, image, video], outputs=video_out, title="Image animation",).launch(share=True)
