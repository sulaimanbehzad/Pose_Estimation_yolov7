import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
import PySimpleGUI as sg
import pickle
import codecs

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
labels2 = ['l_ankle', 'l_knee', 'r_ankle', 'r_knee', 'mid1', 'f1', 'f2', 'f3', 'l_shoulder', 'r_shoulder', 'l_elbow',
           'r_elbow', 'f5', 'f6', 'f7', 'f8', 'uk1', 'uk2', 'uk3']
labels = list('ABCDEFGHIJKLMNOP')
c = np.random.randint(1, 5, size=16)

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]


def plot_points(im_left, im_right, df_left, df_right, win):
    graph = win['-GRAPH-']
    # TODO: delete if doesn't work
    # graph.grab_anywhere_exclude()
    # clear graph to fix issue of overlapping images
    graph.erase()
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
    image_left = image_left.resize((IM_WIDTH // 2, IM_HEIGHT // 2))
    image_right = image_right.resize((IM_WIDTH // 2, IM_HEIGHT // 2))
    bio_left = io.BytesIO()
    bio_right = io.BytesIO()

    image_left.save(bio_left, format="PNG")
    image_right.save(bio_right, format="PNG")
    data1 = bio_left.getvalue()
    data2 = bio_right.getvalue()
    try:
        print('drawing image')
        graph.draw_image(data=data1, location=(0, 0))
        graph.draw_image(data=data2, location=(IM_WIDTH / 2, 0))

    except:
        pass

    vals_left = df_left.loc[df_left['image'] == im_left]
    vals_right = df_right.loc[df_right['image'] == im_right]

    camera_points_left = vals_left[['kpt_x', 'kpt_y']].to_numpy()
    camera_points_right = vals_right[['kpt_x', 'kpt_y']].to_numpy()

    left_xy_coord.append(camera_points_left)
    right_xy_coord.append(camera_points_right)

    for cpl, cpr in zip(camera_points_left, camera_points_right):
        # print(f'cpl: {cpl} - cpr: {cpr}')
        graph.draw_circle(center_location=(cpl[0] / 2, cpl[1] / 2), radius=5, fill_color='black', line_color='red')
        cpr[0] /= 2
        cpr[0] += IM_WIDTH / 2
        graph.draw_circle(center_location=(cpr[0], cpr[1] / 2), radius=5, fill_color='black', line_color='red')
    # kpts = left_camera_points
    # steps = 2
    # for sk_id, sk in enumerate(skeleton):
    #     r, g, b = pose_limb_color[sk_id]
    #     pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
    #     pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
    #     if steps == 3:
    #         conf1 = kpts[(sk[0]-1)*steps+2]
    #         conf2 = kpts[(sk[1]-1)*steps+2]
    #         if conf1<0.5 or conf2<0.5:
    #             continue
    #     if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
    #         continue
    #     if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
    #         continue
    #     graph.line(pos1, pos2, (int(r), int(g), int(b)), thickness=2)
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


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def draw_image_pair(im_left, im_right, graph):
    image_left = Image.open(im_left)
    image_right = Image.open(im_right)

    image_left = image_left.resize((IM_WIDTH // 2, IM_HEIGHT // 2))
    image_right = image_right.resize((IM_WIDTH // 2, IM_HEIGHT // 2))

    bio_left = io.BytesIO()
    bio_right = io.BytesIO()

    image_left.save(bio_left, format="PNG")
    image_right.save(bio_right, format="PNG")
    data1 = bio_left.getvalue()
    data2 = bio_right.getvalue()
    try:
        print('drawing image')
        graph.draw_image(data=data1, location=(0, 0))
        graph.draw_image(data=data2, location=(IM_WIDTH / 2, 0))

    except Exception as inst:
        print(inst)


def draw_image_details(im_left, im_right, df_im_left, df_right, win):
    graph = win['-GRAPH-']
    graph.erase()
    left_xy_coord = []
    right_xy_coord = []
    draw_image_pair(im_left, im_right, graph)

    vals_left = df_im_left.loc[df_im_left['image'] == im_left]
    vals_right = df_right.loc[df_right['image'] == im_right]

    output_left = pickle.loads(codecs.decode(vals_left.iloc[0]['output'].encode(), "base64"))
    output_right = pickle.loads(codecs.decode(vals_right.iloc[0]['output'].encode(), "base64"))
    # print(f'vals left: \n{output_left} \nvalse right: \n{output_right}')
    # output_right = pickle.loads(codecs.decode(vals_right['output'].encode(), "base64"))
    # print(f'len output: {len(output_left)} \nleft output: {output_left}')
    steps = 3  # where's the next datapoint located
    num_kpts = len(output_left) // steps
    for kid in range(num_kpts):
        x_coord, y_coord = output_left[steps * kid], output_left[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = output_left[steps * kid + 2]
                if conf < 0.5:
                    continue
        left_xy_coord.append([int(x_coord), int(y_coord)])

    num_kpts = len(output_right) // steps
    for kid in range(num_kpts):
        x_coord, y_coord = output_right[steps * kid], output_right[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = output_right[steps * kid + 2]
                if conf < 0.5:
                    continue
        right_xy_coord.append([int(x_coord), int(y_coord)])

    for i in range(1, 16):
        point = point_to_gui_coordinate(left_xy_coord[i-1], True)
        window[f'txt{i}'].update(str(point))
    draw_skeleton_2D(output_left, left_xy_coord, 3, True)
    draw_skeleton_2D(output_right, right_xy_coord, 3, False)
    for cpl, cpr in zip(left_xy_coord, right_xy_coord):
        print(f'cpl: {cpl} - cpr: {cpr}')
        graph.draw_circle(center_location=point_to_gui_coordinate(cpl, True), radius=4, fill_color='yellow', line_color='red')
        graph.draw_circle(center_location=point_to_gui_coordinate(cpr, False), radius=4, fill_color='yellow', line_color='red')
    return left_xy_coord, right_xy_coord

def point_to_gui_coordinate(point, is_left):
    ret_point = [-1, -1]
    if is_left:
        ret_point[0] = point[0] / 2
    else:
        ret_point[0] = point[0]/2
        ret_point[0] += IM_WIDTH/2
    ret_point[1] = point[1]
    return ret_point


def draw_skeleton_2D(kpts, coords_and_label, steps, is_left):
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        raw_pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        raw_pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        if is_left:
            pos1 = (int(kpts[(sk[0] - 1) * steps] // 2), int(kpts[(sk[0] - 1) * steps + 1]))
            pos2 = (int(kpts[(sk[1] - 1) * steps] // 2), int(kpts[(sk[1] - 1) * steps + 1]))
        else:
            pos1 = (int(kpts[(sk[0] - 1) * steps] // 2) + (IM_WIDTH / 2), int(kpts[(sk[0] - 1) * steps + 1]))
            pos2 = (int(kpts[(sk[1] - 1) * steps] // 2) + (IM_WIDTH / 2), int(kpts[(sk[1] - 1) * steps + 1]))
        if steps == 3:
            conf1 = kpts[(sk[0] - 1) * steps + 2]
            conf2 = kpts[(sk[1] - 1) * steps + 2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
        if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        hex_color = rgb_to_hex((r, g, b))
        hex_color = '#' + hex_color
        # print(f'RGB: {r} {g} {b} \n  hex: {hex_color}')
        graph.draw_line(pos1, pos2, hex_color, width=4)
        # print(f'drew line from  {pos1} to {pos2}')
        # print(f'raw pos1 {raw_pos1}')
        # TODO: what if the label is -1
        if sk_id != -1:
            graph.draw_text(str(labels2[sk_id]), pos1, color='black')
            # graph.Multiline(default_text=str(labels2[sk_id]), size=(30, 5), key=str(sk_id))
        print(f'coords {coords_and_label} type coords {type(coords_and_label)}')
        try:
            ind_to_add = coords_and_label.index([raw_pos1[0], raw_pos1[1]])
            # print(f'{ind_to_add}')
            coords_and_label[ind_to_add].append(sk_id)
        except:
            pass
        print(f'coords after {coords_and_label} type coords {type(coords_and_label)}')


def pixel_to_world(camera_intrinsics, r, t, img_points):
    K_inv = camera_intrinsics.I
    R_inv = np.asmatrix(r).I
    R_inv_T = np.dot(R_inv, np.asmatrix(t))
    world_points = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)
        cam_R_inv = np.dot(R_inv, cam_point)
        scale = R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0] = world_point[0]
        pt[1] = world_point[1]
        pt[2] = 0
        world_points.append(pt.T.tolist())

    return world_points


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
    # print('plot 3d')
    p3ds = []
    for itr1, itr2 in zip(lkpts, rkpts):
        itr1, itr2 = combine_labels(itr1, itr2)
        _p3d = DLT(P1, P2, itr1[0:2], itr2[0:2])
        if len(itr1) < 3:
            _p3d = np.append(_p3d, -1)
        else:
            _p3d = np.append(_p3d, itr2[2])
        p3ds = np.append(p3ds, _p3d)
        # p3ds = np.array(p3ds)
    # print(f'len p3ds {len(p3ds)}')
    p3ds = np.reshape(p3ds, (16, 4))
    # for p in p3ds:
    #     print(f'3d {p}')

    from mpl_toolkits.mplot3d import Axes3D

    # min_thresh = np.min(p3ds)
    # max_thresh = np.max(p3ds)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(np.min(p3ds[0]), np.max(p3ds[0]))
    ax.set_ylim3d(np.min(p3ds[1]), np.max(p3ds[1]))
    ax.set_zlim3d(np.min(p3ds[2]), np.max(p3ds[2]))
    prev_p = []
    for p, l in zip(p3ds, lkpts):
        ax.scatter(xs=p[0], ys=p[1], zs=p[2], c='red', s=3)
        ax.text(p[0], p[1], p[2], (str(labels2[int(p[3])])), size=5, zorder=1, color='k')

        # if len(prev_p) > 0:
        #     ax.plot([prev_p[0], p[0]], [prev_p[1], p[1]], [prev_p[2], p[2]], color='black')
        # prev_p = p
    # show origin point
    ax.scatter(0, 0, 0, c='black', s=5)


def XYZ_coords_to_csv(left_points, right_points, P1, P2, output_path):
    df = pd.DataFrame(columns=['image_index', 'kpt_x', 'kpt_y', 'kpt_z', 'label'])
    image_num = 1
    for itr_im1, itr_im2 in zip(left_points, right_points):
        for itr1, itr2 in zip(itr_im1, itr_im2):
            itr1, itr2 = combine_labels(itr1, itr2)
            p3ds = []
            # print(f'=========== {itr1[0:2]}')
            _p3d = DLT(P1, P2, itr1[0:2], itr2[0:2])
            if len(itr1) < 3:
                _p3d = np.append(_p3d, -1)
            else:
                _p3d = np.append(_p3d, itr1[2])
            p3ds.append(_p3d)
            p3ds = np.array(p3ds)
            for p in p3ds:
                if p[3] != -1:
                    df2 = pd.DataFrame.from_records([{'image_index': image_num, 'kpt_x': int(p[0]), 'kpt_y': int(p[1]),
                                                      'kpt_z': int(p[2]), 'label': labels2[int(float(p[3]))]}])
                else:
                    df2 = pd.DataFrame.from_records([{'image_index': image_num, 'kpt_x': int(p[0]), 'kpt_y': int(p[1]),
                                                      'kpt_z': int(p[2]), 'label': int(p[3])}])
                df = pd.concat([df, df2])
        image_num += 1
    df.to_csv(output_path, mode='w+', index=False)


def combine_labels(left_prop, right_prop):
    print(f'left point: {left_prop} right point: {right_prop}')
    if len(left_prop) > len(right_prop):
        right_prop.append(left_prop[2])
    elif len(left_prop) < len(right_prop):
        left_prop.append(right_prop[2])

    print(f'AFTER\nleft point: {left_prop} right point: {right_prop}')

    return left_prop, right_prop


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

    # for l in labels:
    #     print(l)
    # Read csv files
    df_left = pd.read_csv(left_kpts)
    df_right = pd.read_csv(right_kpts)
    imgs_left = df_left.loc[:, 'image'].drop_duplicates()
    imgs_right = df_right.loc[:, 'image'].drop_duplicates()

    sg.theme('Dark Blue 3')
    controls_column = [
        [sg.Text("Controls: ")], [sg.Button('show image', enable_events=True, key="-SHOW-"),
        sg.Button('exit', key='-EXIT-'),
        sg.R('Move Stuff', 1, key='-MOVE-', enable_events=True), sg.InputText('Select a label to edit', key='-LABEL-')]
    ]
    labels_column = [[sg.Text(f'{i}. ', key=f'txt{i}', size=(10,1)), sg.Multiline(f"{i} txt", key=f'input{i}', size=(20,1)),
                      sg.Text(f'{i+1}. ', key=f'txt{i+1}', size=(10,1)), sg.Multiline(f"{i+1} txt", key=f'input{i+1}', size=(20,1))] for i in range(1, 16, 2)]
    layout = [
        [sg.Text(key='-INFO-', size=(60, 1))],
        [sg.Graph(
            canvas_size=(IM_WIDTH, IM_HEIGHT//2),
            graph_bottom_left=(0, IM_HEIGHT),
            graph_top_right=(IM_WIDTH, 0),
            key="-GRAPH-",
            background_color='lightblue',
            enable_events=True,
            drag_submits=True
        )],
        [sg.Column(labels_column)],
        [sg.Column(controls_column)],


    ]


    window = sg.Window(title='Keypoint Editor', layout=layout, size=(1080, 720),
                       element_padding=5, location=(0, 0))

    itr_left = 0
    itr_right = 0

    window.finalize()
    window.maximize()
    graph = window['-GRAPH-']
    dragging = False
    start_point = end_point = prior_rect = None
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "-EXIT-" or event == sg.WIN_CLOSED:
            break

        if event in ('-MOVE-', '-MOVEALL-'):
            graph.set_cursor(cursor='fleur')  # not yet released method... coming soon!
        elif not event.startswith('-GRAPH-'):
            graph.set_cursor(cursor='left_ptr')  # not yet released method... coming soon!

        if event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
            x, y = values["-GRAPH-"]
            if not dragging:
                start_point = (x, y)
                dragging = True
                drag_figures = graph.get_figures_at_location((x, y))
                drag_figures = tuple([d for d in drag_figures if d != 1 if d != 2])
                # window['-LABEL-'].update(value=drag_figures[0])
                for d in drag_figures:
                    print(f'figure {d}')
                lastxy = x, y
            else:
                end_point = (x, y)
            if prior_rect:
                graph.delete_figure(prior_rect)
            delta_x, delta_y = x - lastxy[0], y - lastxy[1]
            lastxy = x, y
            if None not in (start_point, end_point):
                if values['-MOVE-']:
                    for fig in drag_figures:
                        graph.move_figure(fig, delta_x, delta_y)
                        graph.update()
            window["-INFO-"].update(value=f"mouse {values['-GRAPH-']}")
        elif event.endswith('+UP'):  # The drawing has ended because mouse up
            window["-INFO-"].update(value=f"grabbed rectangle from {start_point} to {end_point}")
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
            prior_rect = None
        if event == '-SHOW-':
            if (itr_left and itr_right) < len(imgs_left):
                lfp, rfp = draw_image_details(imgs_left.iloc[itr_left], imgs_right.iloc[itr_right], df_left, df_right,
                                              window)
                left_camera_points.append(lfp)
                right_camera_points.append(rfp)
                # combine_labels(left_camera_points, right_camera_points)
                itr_left += 1
                itr_right += 1
    # print(f'size lfps: {len(left_camera_points)}')
    # print(left_camera_points)
    XYZ_coords_to_csv(left_camera_points, right_camera_points, projection_matrix_1, projection_matrix_2,
                      'data/out/XYZ_Coords.csv')
    plot_keypoints_3d(left_camera_points[1], right_camera_points[1], projection_matrix_1, projection_matrix_2)
    plt.show()
    plt.waitforbuttonpress()
    window.close()
