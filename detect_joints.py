import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tkinter
import matplotlib
import os
from Calibrate_Multiple_Cameras import SortImageNames

matplotlib.use('TkAgg')

print(torch.version)

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model

model = load_model()


def run_inference(url):
    image = cv2.imread(url) # shape: (480, 640, 3)
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 768, 960])
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 768, 960])
    output, _ = model(image) # torch.Size([1, 45900, 57])
    return output, image

def visualize_output(output, image):
    output = non_max_suppression_kpt(output,
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    print(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(nimg)
    plt.show()

def multiple_camera_keypoint():
    left_imgs_path = 'data/pose_imgs/LeftCamera'
    right_imgs_path = 'data/pose_imgs/RightCamera'
    print('We have {} Images from the left camera'.format(len(os.listdir(left_imgs_path))))
    print('and {} Images from the right camera.'.format(len(os.listdir(right_imgs_path))))

    print('Before: {}, {}, {}, ...'.format(os.listdir(left_imgs_path)[0], os.listdir(left_imgs_path)[1], os.listdir(left_imgs_path)[2]))
    left_imgs = SortImageNames(left_imgs_path)
    right_imgs = SortImageNames(right_imgs_path)
    print('After: {}, {}, {}, ...'.format(os.path.basename(left_imgs[0]), os.path.basename(left_imgs[1]), os.path.basename(left_imgs[2])))
    print(f'{len(left_imgs)} \n{len(right_imgs)}')
    
multiple_camera_keypoint()
# imgs_path = 'images/IM_L_11.jpg'
# output, image = run_inference(imgs_path) # Bryan Reyes on Unsplash
# visualize_output(output, image)
