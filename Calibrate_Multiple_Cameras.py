import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import time

# link for obtaining chessboard pattern https://calib.io/pages/camera-calibration-pattern-generator
# check if opencv is installed
print(cv2.__version__)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)


def SortImageNames(path):
    imagelist = sorted(os.listdir(path))
    lengths = []
    for name in imagelist:
        lengths.append(len(name))
    lengths = sorted(list(set(lengths)))
    ImageNames, ImageNamesRaw = [], []
    for l in lengths:
        for name in imagelist:
            if len(name) == l:
                ImageNames.append(os.path.join(path, name))
                ImageNamesRaw.append(name)
    return ImageNames


def GenerateImagepoints(paths, board_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgpoints = []
    for name in paths:
        img = cv2.imread(name)
        img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        # print(f'width: {width} - {height}')
        temp = gray
        # if height > 1000:
        #     temp = cv2.GaussianBlur(gray, (0, 0), cv2.BORDER_DEFAULT)
        #     temp = cv2.addWeighted(gray, 1.8, temp, -0.8,0,gray)
        ret, corners1 = cv2.findChessboardCorners(temp, board_size)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners1, (4, 4), (-1, -1), criteria)
            imgpoints.append(corners2)
    return imgpoints



# we also can display the imagepoints on the example pictures.

def DisplayImagePoints(path, imgpoints, board_size):
    img = cv2.imread(path)
    img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.drawChessboardCorners(img, board_size, imgpoints, True)
    return img



# cv2.imshow('im', example_image_left)
# cv2.waitKey(10000)

# in this picture we now see the local coordinate system of the chessboard
# the origin is at the top left corner
# the orientation is like: long side = X

def PlotLocalCoordinates(img, points, board_size):
    points = np.int32(points)
    cv2.arrowedLine(img, tuple(points[0, 0]), tuple(points[3, 0]), (255, 0, 0), 3, tipLength=0.05)
    cv2.arrowedLine(img, tuple(points[0, 0]), tuple(points[board_size[0] * 3, 0]), (255, 0, 0), 3, tipLength=0.05)
    cv2.circle(img, tuple(points[0, 0]), 8, (0, 255, 0), 3)
    cv2.putText(img, '0,0', (points[0, 0, 0] - 35, points[0, 0, 1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, 'X', (points[4, 0, 0] - 25, points[4, 0, 1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                2, cv2.LINE_AA)
    cv2.putText(img, 'Y', (points[board_size[0] * 3, 0, 0] - 25, points[board_size[0] * 3, 0, 1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img




def CalibrateCamera(paths, imgpoints, objpoints):
    CameraParams = {}
    temp = cv2.imread(paths[0])
    temp = cv2.resize(temp, (IM_WIDTH, IM_HEIGHT))
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    g = gray.shape[::-1]

    flags = 0

    objp = []
    for i in range(len(imgpoints)):
        objp.append(objpoints)
    (ret, mtx, dist, rvecs, tvecs) = cv2.calibrateCamera(objp, imgpoints, g, None, None, flags=flags)

    Rmtx = []
    Tmtx = []
    k = 0
    for r in rvecs:
        Rmtx.append(cv2.Rodrigues(r)[0])
        Tmtx.append(np.vstack((np.hstack((Rmtx[k], tvecs[k])), np.array([0, 0, 0, 1]))))
        k += 1

    img = cv2.imread(paths[0], 0)
    img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))
    h, w = img.shape[:2]
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    if np.sum(roi) == 0:
        roi = (0, 0, w - 1, h - 1)

    CameraParams['Intrinsic'] = mtx
    CameraParams['Distortion'] = dist
    CameraParams['DistortionROI'] = roi
    CameraParams['DistortionIntrinsic'] = newmtx
    CameraParams['RotVektor'] = rvecs
    CameraParams['RotMatrix'] = Rmtx
    CameraParams['Extrinsics'] = Tmtx
    CameraParams['TransVektor'] = tvecs

    return CameraParams





def CalculateErrors(params, imgpoints, objpoints):
    imgp = np.array(imgpoints)
    imgp = imgp.reshape((imgp.shape[0], imgp.shape[1], imgp.shape[3]))
    objp = np.array(objpoints)
    K = np.array(params['Intrinsic'])
    D = np.array(params['Distortion'])
    R = np.array(params['RotVektor'])
    T = np.array(params['TransVektor'])
    N = imgp.shape[0]

    imgpNew = []
    for i in range(N):
        temp, _ = cv2.projectPoints(objp, R[i], T[i], K, D)
        imgpNew.append(temp.reshape((temp.shape[0], temp.shape[2])))
    imgpNew = np.array(imgpNew)

    err = []
    for i in range(N):
        err.append(imgp[i] - imgpNew[i])
    err = np.array(err)

    def RMSE(err):
        return np.sqrt(np.mean(np.sum(err ** 2, axis=1)))

    errall = np.copy(err[0])
    rmsePerView = [RMSE(err[0])]
    for i in range(1, N):
        errall = np.vstack((errall, err[i]))
        rmsePerView.append(RMSE(err[i]))

    rmseAll = RMSE(errall)
    return rmsePerView, rmseAll





def StereoCalibration(leftparams, rightparams, objpoints, imgpL, imgpR, Left_Paths):
    StereoParams = {}

    k1 = leftparams['Intrinsic']
    d1 = leftparams['Distortion']
    k2 = rightparams['Intrinsic']
    d2 = rightparams['Distortion']
    gray = cv2.imread(Left_Paths[0], 0)
    gray = cv2.resize(gray, (IM_WIDTH, IM_HEIGHT))
    g = gray.shape[::-1]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    objp = []
    for i in range(len(imgpL)):
        objp.append(objpoints)

    (ret, K1, D1, K2, D2, R, t, E, F) = cv2.stereoCalibrate(objp, imgpL, imgpR, k1, d1, k2, d2, g, criteria=criteria,
                                                            flags=flags)

    T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))

    StereoParams['Transformation'] = T
    StereoParams['t'] = t
    StereoParams['R'] = R
    StereoParams['Essential'] = E
    StereoParams['Fundamental'] = F
    StereoParams['MeanError'] = ret
    return StereoParams

def run_calibration(left_camera_dir, right_camera_dir, board_size, square_size):
    print('We have {} Images from the left camera'.format(len(os.listdir(left_camera_dir))))
    print('and {} Images from the right camera.'.format(len(os.listdir(right_camera_dir))))

    # sort the image names after their number
    # save the image names with the whole path in a list

    print(
        'Before: {}, {}, {}, ...'.format(os.listdir(left_camera_dir)[0], os.listdir(left_camera_dir)[1], os.listdir(left_camera_dir)[2]))

    Left_Path_Sorted = SortImageNames(left_camera_dir)
    Right_Path_Sorted = SortImageNames(right_camera_dir)

    print('After: {}, {}, {}, ...'.format(os.path.basename(Left_Path_Sorted[0]), os.path.basename(Left_Path_Sorted[1]),
                                          os.path.basename(Left_Path_Sorted[2])))

    # we have to create the objectpoints
    # that are the local 2D-points on the pattern, corresponding
    # to the local coordinate system on the top left corner.

    objpoints = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objpoints[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objpoints *= square_size

    # now we have to find the imagepoints
    # these are the same points like the objectpoints but depending
    # on the camera coordination system in 3D
    # the imagepoints are not the same for each image/camera

    Left_imgpoints = GenerateImagepoints(Left_Path_Sorted, board_size)
    Right_imgpoints = GenerateImagepoints(Right_Path_Sorted, board_size)
    print(f'Detected left: {len(Left_imgpoints)}')
    print(f'Detected right: {len(Left_imgpoints)}')

    example_image_left = DisplayImagePoints(Left_Path_Sorted[0], Left_imgpoints[0], board_size)
    example_image_right = DisplayImagePoints(Right_Path_Sorted[0], Right_imgpoints[0], board_size)
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

    for ax, im in zip(grid, [example_image_left, example_image_right]):
        ax.imshow(im)
        ax.axis('off')

    n = 2
    img = cv2.imread(Left_Path_Sorted[n])
    img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('ex', img)
    # img = PlotLocalCoordinates(img, Left_imgpoints[n])

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(30)
    plt.close()

    # Camera Calibration
    Start_Time_Cal = time.perf_counter()

    Left_Params = CalibrateCamera(Left_Path_Sorted, Left_imgpoints, objpoints)
    Right_Params = CalibrateCamera(Right_Path_Sorted, Right_imgpoints, objpoints)

    np.set_printoptions(suppress=True, precision=5)
    print('Intrinsic Matrix:')
    print(Left_Params['Intrinsic'])
    print('\nDistortion Parameters:')
    print(Left_Params['Distortion'])
    print('\nExtrinsic Matrix from 1.Image:')
    print(Left_Params['Extrinsics'][0])

    # stuff only to run when not called via 'import' here
    Left_Errors, Left_MeanError = CalculateErrors(Left_Params, Left_imgpoints, objpoints)
    Right_Errors, Right_MeanError = CalculateErrors(Right_Params, Right_imgpoints, objpoints)

    print('Reprojection Error Left:  {:.4f}'.format(Left_MeanError))
    print('Reprojection Error Right: {:.4f}'.format(Right_MeanError))

    Left_Params['Imgpoints'] = Left_imgpoints
    Left_Params['Errors'] = Left_Errors
    Left_Params['MeanError'] = Left_MeanError

    Right_Params['Imgpoints'] = Right_imgpoints
    Right_Params['Errors'] = Right_Errors
    Right_Params['MeanError'] = Right_MeanError

    Stereo_Params = StereoCalibration(Left_Params, Right_Params, objpoints, Left_imgpoints, Right_imgpoints, Left_Path_Sorted)

    print('Transformation Matrix:')
    print(Stereo_Params['Transformation'])
    print('\nEssential Matrix:')
    print(Stereo_Params['Essential'])
    print('\nFundamental Matrix:')
    print(Stereo_Params['Fundamental'])
    print('\nMean Reprojection Error:')
    print('{:.6f}'.format(Stereo_Params['MeanError']))

    end = time.perf_counter() - Start_Time_Cal
    print('elapsed time for calibration process: {:.2f} seconds.'.format(end))

    Parameters = Stereo_Params
    Parameters['SquareSize'] = square_size
    Parameters['BoardSize'] = board_size
    Parameters['Objpoints'] = objpoints

    for Lkey in Left_Params.keys():
        name = 'L_' + str(Lkey)
        Parameters[name] = Left_Params[Lkey]

    for Rkey in Right_Params.keys():
        name = 'R_' + str(Rkey)
        Parameters[name] = Right_Params[Rkey]

    # save the Parameters dictionary into an npz file
    # with this file we can access the data afterwards very easy

    file = 'data/out/parameters.npz'
    np.savez(file, **Parameters)
    npz = dict(np.load(file))
    size = (npz['L_Imgpoints'].shape[0], npz['L_Imgpoints'].shape[1], npz['L_Imgpoints'].shape[3])
    npz['L_Imgpoints'] = np.resize(npz.pop('L_Imgpoints'), size)
    npz['R_Imgpoints'] = np.resize(npz.pop('R_Imgpoints'), size)
    np.savez(file, **npz)

# stuff to run always here such as class/def
def main():
    pass


if __name__ == "__main__":

    # Current Grid 11 * 7
    # New Grid 10 * 4
    # New Grid 10 * 6
    # New Grid 8 * 4
    # New Grid 9 * 3
    BOARD_SIZE = (4, 8)

    IM_HEIGHT = 576
    IM_WIDTH = 1024
    # square size
    SQUARE_SIZE = 12.5

    # Images directory for loading
    LEFT_PATH = 'data/calib_imgs5/leftcamera'
    RIGHT_PATH = 'data/calib_imgs5/rightcamera'
    run_calibration(LEFT_PATH, RIGHT_PATH, BOARD_SIZE, SQUARE_SIZE)
    main()
