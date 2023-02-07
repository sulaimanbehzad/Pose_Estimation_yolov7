import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

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
           'l_wrist', 'mid1', 'mid2', 'r_wrist', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
labels = list('ABCDEFGHIJKLMNOP')
for l in labels:
    print(l)

c = np.random.randint(1,5,size=16)

def plot_points(im_dir):
    df = pd.read_csv(im_dir)
    imgs = df.loc[:, 'image'].drop_duplicates()
    # print(imgs)
    xy_coord = []
    for im in imgs:
        fig, ax = plt.subplots()
        frame1 = plt.imread(im, format='jpg')
        frame1 = cv.resize(frame1, (IM_WIDTH, IM_HEIGHT))
        imagebox = OffsetImage(frame1, zoom=0.2)
        imagebox.image.axes = ax
        vals = df.loc[df['image'] == im]
        camera_points = vals[['kpt_x', 'kpt_y']].to_numpy()
        xy_coord.append(camera_points)
        plt.imshow(frame1[:, :, [2, 1, 0]])
        sc = plt.scatter(camera_points[:, 0], camera_points[:, 1], c='r', s=4)
        print(f'{camera_points[:,0]} ----------------- {camera_points[:, 1]}')
        for txt,x_coord, y_coord in zip(labels2, camera_points[:, 0], camera_points[:, 1]):
            ax.annotate(txt, (x_coord, y_coord), color='b')
        plt.show()  # left_camera_points = np.array(left_camera_points.va)
    return xy_coord
def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()
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

left_camera_points = plot_points(left_kpts)
right_camera_points = plot_points(right_kpts)

print(left_camera_points)

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


plot_keypoints_3d(left_camera_points, right_camera_points, projection_matrix_1, projection_matrix_2)

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
