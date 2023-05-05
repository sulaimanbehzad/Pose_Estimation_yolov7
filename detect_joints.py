import pandas as pd
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
from GPUtil import showUtilization as gpu_usage
import gc
import pickle
import codecs

matplotlib.use('TkAgg')

print(torch.version)

IM_HEIGHT = 576
IM_WIDTH = 1024


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()
    gc.collect()
    torch.cuda.empty_cache()

    # cuda.select_device(0)
    # cuda.close()
    # cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


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
    free_gpu_cache()
    image = cv2.imread(url)  # shape: (480, 640, 3)
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0]  # shape: (768, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image)  # torch.Size([3, 768, 960])
    # Turn image into batch
    image = image.unsqueeze(0)  # torch.Size([1, 3, 768, 960])
    if torch.cuda.is_available():
        print('send image to cuda')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image = image.to(device)
    # the two with statements code runs on GPU
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            output, _ = model(image)  # torch.Size([1, 45900, 57])
    return output, image


def visualize_output(output, image):
    output = non_max_suppression_kpt(output,
                                     0.25,  # Confidence Threshold
                                     0.65,  # IoU Threshold
                                     nc=model.yaml['nc'],  # Number of Classes
                                     nkpt=model.yaml['nkpt'],  # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    print(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        print(f'keypoints for skeleton: {output[idx, 7:].T}')  # has dimension: 48 : 16 x, 16 y, 16 threshold
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(nimg)
    plt.show()


def multiple_pics_inference(path):
    output = []
    image = []
    print('Before: {}, {}, {}, ...'.format(os.listdir(path)[0], os.listdir(path)[1], os.listdir(path)[2]))
    sorted_path = SortImageNames(path)
    print('After: {}, {}, {}, ...'.format(os.path.basename(path[0]), os.path.basename(path[1]),
                                          os.path.basename(path[2])))
    for p in sorted_path:
        print(f'{p}')
        out, im = run_inference(p)
        output.append(out)
        image.append(im)
    return output, image


def visualize_multiple_pics(output, image):
    for (out, im) in zip(output, image):
        visualize_output(out, im)


def get_keypoints(output, image):
    output = non_max_suppression_kpt(output,
                                     0.25,  # Confidence Threshold
                                     0.65,  # IoU Threshold
                                     nc=model.yaml['nc'],  # Number of Classes
                                     nkpt=model.yaml['nkpt'],  # Number of Keypoints
                                     kpt_label=True)
    # TODO: get keypoint label and save it in dataframe
    with torch.no_grad():
        output = output_to_keypoint(output)
    print(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_kpts(nimg, output[idx, 7:].T, 3)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(nimg)
    plt.show()


def plot_kpts(im, kpts, steps, orig_shape=None):
    num_kpts = len(kpts) // steps
    radius = 5
    for kid in range(num_kpts):
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (255, 0, 0), -1)


def save_kpts(imgs_path, out_path, orig_shape=None):
    df = pd.DataFrame(columns=['image', 'kpt_x', 'kpt_y'])
    # 'keypoint_label'
    for img_p in imgs_path:
        output, _ = run_inference(img_p)
        output = non_max_suppression_kpt(output,
                                         0.25,  # Confidence Threshold
                                         0.65,  # IoU Threshold
                                         nc=model.yaml['nc'],  # Number of Classes
                                         nkpt=model.yaml['nkpt'],  # Number of Keypoints
                                         kpt_label=True)
        # DONE: get keypoint label and save it in dataframe
        with torch.no_grad():
            output = output_to_keypoint(output)
        # print(f'output: {output}')
        for idx in range(output.shape[0]):
            kpts = output[idx, 7:].T
            steps = 3  # where's the next datapoint located
            num_kpts = len(kpts) // steps
            for kid in range(num_kpts):
                x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
                if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                    if steps == 3:
                        conf = kpts[steps * kid + 2]
                        if conf < 0.5:
                            continue
                    # cv2.circle(im, (int(x_coord), int(y_coord)), radius, (255,0,0), -1)
                    print(f'x: {x_coord} - y: {y_coord}')
                    df2 = pd.DataFrame.from_records([{'image': img_p, 'kpt_x': int(x_coord), 'kpt_y': int(y_coord)}])
                    # print(f'{df2}')
                    # print(f'df before concat: {df}')
                    df = pd.concat([df, df2])
                    # print(f'df after concat: {df}')
    df.to_csv(out_path, mode='w+', index=False)


def save_keypoint_with_id(imgs_path, out_path, orig_shape=None):
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    steps = 3

    df = pd.DataFrame(columns=['image', 'kpt_x', 'kpt_y', 'sk_id'])
    # 'keypoint_label'
    for img_p in imgs_path:
        output, _ = run_inference(img_p)
        output = non_max_suppression_kpt(output,
                                         0.25,  # Confidence Threshold
                                         0.65,  # IoU Threshold
                                         nc=model.yaml['nc'],  # Number of Classes
                                         nkpt=model.yaml['nkpt'],  # Number of Keypoints
                                         kpt_label=True)
        # TODO: get keypoint label and save it in dataframe
        with torch.no_grad():
            output = output_to_keypoint(output)
        # print(f'output: {output}')
        for idx in range(output.shape[0]):
            kpts = output[idx, 7:].T
            steps = 3  # where's the next datapoint located
            num_kpts = len(kpts) // steps
            for kid in range(num_kpts):
                x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
                if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                    if steps == 3:
                        conf = kpts[steps * kid + 2]
                        if conf < 0.5:
                            continue
                    # cv2.circle(im, (int(x_coord), int(y_coord)), radius, (255,0,0), -1)
                    print(f'x: {x_coord} - y: {y_coord}')
                    df2 = pd.DataFrame.from_records([{'image': img_p, 'kpt_x': int(x_coord), 'kpt_y': int(y_coord)}])
                    # print(f'{df2}')
                    # print(f'df before concat: {df}')
                    df = pd.concat([df, df2])
                    # print(f'df after concat: {df}')
            for sk_id, sk in enumerate(skeleton):
                r, g, b = pose_limb_color[sk_id]
                pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
                pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
                if steps == 3:
                    conf1 = kpts[(sk[0] - 1) * steps + 2]
                    conf2 = kpts[(sk[1] - 1) * steps + 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                # add this line to check ids of skeleton +' sk ' + str(sk)
                # print(f'x values {df.kpt_x}')
                df["sk_id"] = np.where(((df['kpt_x'] == pos1[0]) & (df['kpt_y'] == pos1[1])), sk_id, df['sk_id'])
                df["sk_id"] = np.where(((df['kpt_x'] == pos2[0]) & (df['kpt_y'] == pos2[1])), sk_id, df['sk_id'])

                # cv2.putText(im, 'id ' + str(sk_id), pos1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 50, 255), 2)

    df.to_csv(out_path, mode='w+', index=False)


def save_raw_output(imgs_path, out_path, orig_shape=None):
    df = pd.DataFrame(columns=['image', 'output'])
    # 'keypoint_label'
    for img_p in imgs_path:
        output, _ = run_inference(img_p)
        output = non_max_suppression_kpt(output,
                                         0.25,  # Confidence Threshold
                                         0.65,  # IoU Threshold
                                         nc=model.yaml['nc'],  # Number of Classes
                                         nkpt=model.yaml['nkpt'],  # Number of Keypoints
                                         kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        # print(f'output: {output}')
        for idx in range(output.shape[0]):
            kpts = output[idx, 7:].T
            pickled = codecs.encode(pickle.dumps(kpts), "base64").decode()
            print(f'kpts {pickled}')
            print(
                f'len output: {len(output)} \n type output: {type(output)} \n output: {output} \n type kpts: {type(pickled)}')
            df2 = pd.DataFrame.from_records([{'image': img_p, 'output': pickled}])
            df = pd.concat([df, df2])
    df.to_csv(out_path, mode='w+', index=False)


def main():
    pass


if __name__ == '__main__':
    left_imgs_path = 'data/pose_imgs/Pose6/leftcamera'
    right_imgs_path = 'data/pose_imgs/Pose6/rightcamera'
    output_left_keypoints = 'data/out/keypoint_left_06.csv'
    output_right_keypoints = 'data/out/keypoint_right_06.csv'

    print('We have {} Images from the left camera'.format(len(os.listdir(left_imgs_path))))
    print('and {} Images from the right camera.'.format(len(os.listdir(right_imgs_path))))
    print('Before: {}, {}, {}, ...'.format(os.listdir(left_imgs_path)[0], os.listdir(left_imgs_path)[1],
                                           os.listdir(left_imgs_path)[2]))
    left_sorted = SortImageNames(left_imgs_path)
    right_sorted = SortImageNames(right_imgs_path)
    print('After: {}, {}, {}, ...'.format(os.path.basename(left_sorted[0]), os.path.basename(left_sorted[1]),
                                          os.path.basename(left_sorted[2])))
    left_output, left_image = multiple_pics_inference(left_imgs_path)
    right_output, right_image = multiple_pics_inference(right_imgs_path)
    # visualize_multiple_pics(left_output,left_image)
    # get_keypoints(left_output[2], left_image[2])
    visualize_output(left_output[1], left_image[1])
    save_raw_output(left_sorted, out_path=output_left_keypoints)
    save_raw_output(right_sorted, out_path=output_right_keypoints)
    print(f'GPU: {torch.cuda.is_available()}')
    print(f'GPU: {torch.cuda.device_count()}')
    print(f'GPU: {torch.cuda.current_device()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    # imgs_path = 'images/IM_L_11.jpg'
    # output, image = run_inference(imgs_path) # Bryan Reyes on Unsplash
    # visualize_output(output, image)
