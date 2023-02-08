import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
import PySimpleGUI as sg

IM_HEIGHT = 576
IM_WIDTH = 1024
left_kpts = 'data/out/keypoint_left.csv'
right_kpts = 'data/out/keypoint_right.csv'
file = 'data/out/parameters.npz'
# in the dataset I have stored an example parameters file
param_path = 'data/out/parameters.npz'
# the parameters are stored in a NPZ file
# you can import them to a dictionary and access the parameters
params = dict(np.load(param_path))
print(params.keys())
labels2 = ['f1', 'f2', 'f3', 'f4', 'r_shoulder', 'f_shoulder', 'r_elbow', 'l_elbow'
                                                                          'l_wrist', 'mid1', 'mid2', 'r_wrist',
           'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
labels = list('ABCDEFGHIJKLMNOP')
c = np.random.randint(1, 5, size=16)


def plot_points(im_left, im_right, df_left, df_right, win):
    graph = win['-GRAPH-']
    left_xy_coord = []
    right_xy_coord = []
    # print(f'{type(im_dir)}')
    # im = im_dir.head(1)
    # print(f'{im} | {type(im)}')
    # im_dir.drop(im_dir.head(1).index, inplace=True)
    image_left = Image.open(im_left)
    image_right = Image.open(im_right)
    # image_left.thumbnail((IM_HEIGHT, IM_WIDTH))
    # image_right.thumbnail((IM_HEIGHT, IM_WIDTH))
    image_left = image_left.resize((IM_WIDTH//2, IM_HEIGHT//2))
    image_right = image_right.resize((IM_WIDTH//2, IM_HEIGHT//2))
    bio_left = io.BytesIO()
    bio_right = io.BytesIO()

    image_left.save(bio_left, format="PNG")
    image_right.save(bio_right, format="PNG")
    data1 = bio_left.getvalue()
    data2 = bio_right.getvalue()
    try:
        print('drawing image')
        graph.draw_image(data=data1, location=(0, 0))
        graph.draw_image(data=data2, location=(IM_WIDTH/2, 0))

    except:
        pass

    vals_left = df_left.loc[df_left['image'] == im_left]
    vals_right = df_right.loc[df_right['image'] == im_right]

    camera_points_left = vals_left[['kpt_x', 'kpt_y']].to_numpy()
    camera_points_right = vals_right[['kpt_x', 'kpt_y']].to_numpy()

    for cpl, cpr in zip(camera_points_left, camera_points_right):
        print(f'cpl: {cpl} - cpr: {cpr}')
        graph.draw_circle(center_location=(cpl[0]/2, cpl[1]/2), radius=5, fill_color='black', line_color='red')
        cpr[0] /= 2
        cpr[0] += IM_WIDTH/2
        graph.draw_circle(center_location=(cpr[0], cpr[1]/2), radius=5, fill_color='black', line_color='red')

    # for im in imgs:
    #     image = Image.open(im)
    #     image.thumbnail((IM_HEIGHT, IM_WIDTH))
    #     bio = io.BytesIO()
    #     image.save(bio, format="PNG")
    #     win['-IMAGE-'].update(bio.getvalue())
    #
    # fig, ax = plt.subplots()
    # frame1 = plt.imread(im, format='jpg')
    # frame1 = cv.resize(frame1, (IM_WIDTH, IM_HEIGHT))
    # imagebox = OffsetImage(frame1, zoom=0.2)
    # imagebox.image.axes = ax
    # vals = df.loc[df['image'] == im]
    # camera_points = vals[['kpt_x', 'kpt_y']].to_numpy()
    # xy_coord.append(camera_points)
    # plt.imshow(frame1[:, :, [2, 1, 0]])
    # sc = plt.scatter(camera_points[:, 0], camera_points[:, 1], c='r', s=4)
    # print(f'{camera_points[:,0]} ----------------- {camera_points[:, 1]}')
    # for txt,x_coord, y_coord in zip(labels2, camera_points[:, 0], camera_points[:, 1]):
    #     ax.annotate(txt, (x_coord, y_coord), color='b')
    # plt.show()  # left_camera_points = np.array(left_camera_points.va)
    return left_xy_coord, right_xy_coord


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    # print('A: ')
    # print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)

    # print('Triangulated point: ')
    # print(Vh[3, 0:3] / Vh[3, 3])
    return Vh[3, 0:3] / Vh[3, 3]


def plot_keypoints_3d(lkpts, rkpts, P1, P2):
    for itr1, itr2 in zip(lkpts, rkpts):
        p3ds = []

        for lcp, rcp in zip(itr1, itr2):
            _p3d = DLT(P1, P2, lcp, rcp)
            p3ds.append(_p3d)
        p3ds = np.array(p3ds)

        for p in p3ds:
            print(p)

        from mpl_toolkits.mplot3d import Axes3D

        min_thresh = np.min(p3ds)
        max_thresh = np.max(p3ds)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(min_thresh, max_thresh)
        ax.set_ylim3d(min_thresh, max_thresh)
        ax.set_zlim3d(min_thresh, max_thresh)
        prev_p = []
        for p in p3ds:
            ax.scatter(xs=p[0], ys=p[1], zs=p[2], c='red', s=3)
            if len(prev_p) > 0:
                ax.plot([prev_p[0], p[0]], [prev_p[1], p[1]], [prev_p[2], p[2]], color='black')
            prev_p = p


# left_camera_points = plot_points(left_kpts)
# right_camera_points = plot_points(right_kpts)

# print(left_camera_points)

# # left_camera_points = np.array(left_camera_points)
# right_camera_points = np.array(right_camera_points)
#
# frame1 = cv.imread('data/pose_imgs/Pose3/LeftCamera/Im_L_5.jpg')
# frame1 = cv.resize(frame1, (IM_WIDTH, IM_HEIGHT))
# frame2 = cv.imread('data/pose_imgs/Pose3/RightCamera/Im_R_5.jpg')
# frame2 = cv.resize(frame2, (IM_WIDTH, IM_HEIGHT))
#
# plt.imshow(frame1[:, :, [2, 1, 0]])
# plt.scatter(left_camera_points[:, 0], left_camera_points[:, 1])
# plt.show()
#
# plt.imshow(frame2[:, :, [2, 1, 0]])
# plt.scatter(right_camera_points[:, 0], right_camera_points[:, 1])
# plt.show()


# plot_keypoints_3d(left_camera_points, right_camera_points, projection_matrix_1, projection_matrix_2)

# p3ds = []
#
# for lcp, rcp in zip(left_camera_points, right_camera_points):
#     _p3d = DLT(P1, P2, lcp, rcp)
#     p3ds.append(_p3d)
# p3ds = np.array(p3ds)
#
# for p in p3ds:
#     print(p)
#
# from mpl_toolkits.mplot3d import Axes3D
#
# min_thresh = np.min(p3ds)
# max_thresh = np.max(p3ds)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim3d(min_thresh, max_thresh)
# ax.set_ylim3d(min_thresh, max_thresh)
# ax.set_zlim3d(min_thresh, max_thresh)
# for p in p3ds:
#     ax.scatter(xs=p[0], ys=p[1], zs=p[2], c='red')
#             zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]], c='red'))
# connections = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [1, 9], [2, 8], [5, 9], [8, 9], [0, 10]]
# for _c in connections:
#     print(p3ds[_c[0]])
#     print(p3ds[_c[1]])
#     ax.plot(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]], ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
#             zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]], c='red')
plt.show()


def Get3Dfrom2D(List2D, K, R, t, d=1.75):
    # List2D : n x 2 array of pixel locations in an image
    # K : Intrinsic matrix for camera
    # R : Rotation matrix describing rotation of camera frame
    #     w.r.t world frame.
    # t : translation vector describing the translation of camera frame
    #     w.r.t world frame
    # [R t] combined is known as the Camera Pose.

    List2D = np.array(List2D)
    List3D = []
    # t.shape = (3,1)

    for p in List2D:
        # Homogeneous pixel coordinate
        p = np.array([p[0], p[1], 1]).T
        p.shape = (3, 1)
        # print("pixel: \n", p)

        # Transform pixel in Camera coordinate frame
        pc = np.linalg.inv(K) @ p
        # print("pc : \n", pc, pc.shape)

        # Transform pixel in World coordinate frame
        pw = t + (R @ pc)
        # print("pw : \n", pw, t.shape, R.shape, pc.shape)

        # Transform camera origin in World coordinate frame
        cam = np.array([0, 0, 0]).T
        cam.shape = (3, 1)
        cam_world = t + R @ cam
        # print("cam_world : \n", cam_world)

        # Find a ray from camera to 3d point
        vector = pw - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        # print("unit_vector : \n", unit_vector)

        # Point scaled along this ray
        p3D = cam_world + d * unit_vector
        # print("p3D : \n", p3D)
        List3D.append(p3D)

    return List3D


if __name__ == "__main__":
    left_camera_points = []
    right_camera_points = []

    mtx1 = params['L_Intrinsic']
    mtx2 = params['R_Intrinsic']
    R = params['R']
    T = params['t']
    print(f'R: {R} \n T: {T}\n')
    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    projection_matrix_1 = mtx1 @ RT1  # projection matrix for C1
    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    projection_matrix_2 = mtx2 @ RT2  # projection matrix for C2

    for l in labels:
        print(l)
    # Read csv files
    df_left = pd.read_csv(left_kpts)
    df_right = pd.read_csv(right_kpts)
    imgs_left = df_left.loc[:, 'image'].drop_duplicates()
    imgs_right = df_right.loc[:, 'image'].drop_duplicates()

    image_viewer_column = [
        [sg.Text("Here are the images: ")],
        # [sg.Image(key='-IMAGE_LEFT-')],
        # [sg.Image(key='-IMAGE_RIGHT-')]
    ]
    layout = [
        [sg.Graph(
            canvas_size=(IM_WIDTH, IM_HEIGHT),
            graph_bottom_left=(0, IM_HEIGHT),
            graph_top_right=(IM_WIDTH, 0),
            key="-GRAPH-"
        )],
            # [sg.Column(image_viewer_column)],
        [sg.Button('show image', enable_events=True, key="-SHOW-")],
        [sg.Button('exit', key='-EXIT-')]]
    window = sg.Window(title='Keypoint Editor', layout=layout)

    itr_left = 0
    itr_right = 0

    window.finalize()
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "-EXIT-" or event == sg.WIN_CLOSED:
            break
        if event == '-SHOW-':
            if (itr_left and itr_right) < len(imgs_left):
                lfp, rfp = plot_points(imgs_left.iloc[itr_left], imgs_right.iloc[itr_right],
                                       df_left, df_right, window)
                left_camera_points.append(lfp)
                right_camera_points.append(rfp)
                itr_left += 1
                itr_right += 1

            # if len(imgs_right) > 0:
            #     right_camera_points = plot_points(right_kpts, window)
            # image = Image.open('data/pose_imgs/3.jpg')
            # image.thumbnail((IM_HEIGHT, IM_WIDTH))
            # bio = io.BytesIO()
            # image.save(bio, format="PNG")
            # window['-IMAGE-'].update(bio.getvalue())

    window.close()
