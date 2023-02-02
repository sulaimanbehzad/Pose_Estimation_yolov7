import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

IM_HEIGHT = 576
IM_WIDTH = 1024

file = 'data/out/parameters.npz'
# in the dataset I have stored an example parameters file
param_path = 'data/out/parameters.npz'
# the parameters are stored in a NPZ file
# you can import them to a dictionary and access the parameters
params = dict(np.load(param_path))
print(params.keys())

left_camera_points = [
    [335, 259],    [366, 281],  [389, 352], [439, 288],
    # [591, 421],
    #                   [520, 337],
    #                   [465, 298],
    #                   [553, 439]
]

right_camera_points = [
    # [163, 375],
    #                    [189, 387],
    #                    [242, 294],
    #                    [289, 258],

[312, 255],
                       [330, 216],
                       [287, 268],
                       [251, 251]
                       ]

left_camera_points = np.array(left_camera_points)
right_camera_points = np.array(right_camera_points)

frame1 = cv.imread('data/pose_imgs/Pose3/LeftCamera/Im_L_5.jpg')
frame1 = cv.resize(frame1, (IM_WIDTH, IM_HEIGHT))
frame2 = cv.imread('data/pose_imgs/Pose3/RightCamera/Im_R_5.jpg')
frame2 = cv.resize(frame2, (IM_WIDTH, IM_HEIGHT))

plt.imshow(frame1[:, :, [2, 1, 0]])
plt.scatter(left_camera_points[:, 0], left_camera_points[:, 1])
plt.show()

plt.imshow(frame2[:, :, [2, 1, 0]])
plt.scatter(right_camera_points[:, 0], right_camera_points[:, 1])
plt.show()

mtx1 = params['L_Intrinsic']
mtx2 = params['R_Intrinsic']
R = params['R']
T = params['t']
print(f'R: {R} \n T: {T}\n')
# RT matrix for C1 is identity.
RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
P1 = mtx1 @ RT1  # projection matrix for C1
# RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R, T], axis=-1)
P2 = mtx2 @ RT2  # projection matrix for C2


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


p3ds = []

for lcp, rcp in zip(left_camera_points, right_camera_points):
    _p3d = DLT(P1, P2, lcp, rcp)
    p3ds.append(_p3d)
p3ds = np.array(p3ds)

for p in p3ds:
    print(p)

from mpl_toolkits.mplot3d import Axes3D

min_thresh = np.min(p3ds)
max_thresh = np.max(p3ds)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(min_thresh, max_thresh)
ax.set_ylim3d(min_thresh, max_thresh)
ax.set_zlim3d(min_thresh, max_thresh)
for p in p3ds:
    ax.scatter(xs=p[0], ys=p[1], zs=p[2], c='red')
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
