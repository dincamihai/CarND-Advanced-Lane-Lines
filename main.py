# -*- coding: utf-8 -*-

#!/usr/bin/env python

import cv2
import glob
import pickle
import argparse
import matplotlib
import datetime
import uuid
matplotlib.use('svg')
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from collections import defaultdict
from color import abs_sobel_thresh, mag_thresh, dir_threshold
from utils import annotate_image, get_fit, find_peaks, compute_avg_diff


def save_figure(images, fname=None, cmaps=[]):
    f, axs = plt.subplots(1, len(images), figsize=(24, 9))
    if not cmaps:
        cmaps = [None] * len(images)
    f.tight_layout()
    for idx, (ax, image) in enumerate(zip(axs, images)):
        ax.imshow(image, cmap=cmaps[idx])
        ax.set_title('Image %s' % idx, fontsize=50)
    _uid = uuid.uuid4()
    if not fname:
        fname = 'tmp/orig-%s-%s.png' % (_uid, datetime.datetime.now().strftime("%d-%H-%M-%S"))
    f.savefig(fname)


def calibrate(images):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    shape = None

    good_images = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            good_images.append(img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, shape, None, None)

    with open('cal_matrix.p', 'wb') as _f:
        pickle.dump(mtx, _f)
    with open('cal_dist.p', 'wb') as _f:
        pickle.dump(dist, _f)

    return img, mtx, dist


def undistort(orig_images):
    images = []

    mtx, dist = None, None
    with open('cal_matrix.p', 'rb') as _f:
        mtx = pickle.load(_f)
    with open('cal_dist.p', 'rb') as _f:
        dist = pickle.load(_f)

    for _img in orig_images:
        images.append(cv2.undistort(_img, mtx, dist, None, mtx))

    return images


def crop(image, crop_top=0, debug=0):
    mask1= np.zeros_like(image)
    cv2.fillConvexPoly(
        mask1,
        # vertices
        np.array(
            [
                [mask1.shape[1]-100, image.shape[0]],  # bottom right
                [100, mask1.shape[0]],                 # bottom left
                [image.shape[1]//2-300, crop_top],                           # top left
                [image.shape[1]//2+300, crop_top],            # top right
            ]
        ),
        # color
        (255, 255, 255)
    )
    out = cv2.bitwise_and(image, mask1)
    mask2= np.zeros_like(image)
    cv2.fillConvexPoly(
        mask2,
        # vertices
        np.array(
            [
                [mask2.shape[1]-350, image.shape[0]],  # bottom right
                [350, mask2.shape[0]],                 # bottom left
                [140, crop_top+50],                           # top left
                [image.shape[1]-140, crop_top+50],            # top right
            ]
        ),
        # color
        (255, 255, 255)
    )
    # cv2.bitwise_not(mask2, mask2)
    # out = cv2.bitwise_and(out, mask2)
    if debug >= 2:
        plt.imsave('windows.png', out)
        import pdb; pdb.set_trace()
    return out


def compute_binary(
    channel, ksize, thresholds=dict(sobelx=(0, 255), sobely=(0, 255), magnitude=(0, 255), directional=(-np.pi/2, np.pi/2))
):
    # Apply each of the thresholding functions
    sxbinary = abs_sobel_thresh(channel, orient='x', sobel_kernel=ksize, thresh=thresholds['sobelx'])
    sybinary = abs_sobel_thresh(channel, orient='y', sobel_kernel=ksize, thresh=thresholds['sobely'])
    mag_binary = mag_thresh(channel, sobel_kernel=ksize, thresh=thresholds['magnitude'])
    dir_binary = dir_threshold(channel, sobel_kernel=ksize, thresh=thresholds['directional'])

    # Threshold color channel
    s_binary = np.zeros_like(channel)
    s_binary[((sxbinary == 1) & (sybinary == 1)) | ((dir_binary == 1) & (mag_binary == 1))] = 1

    return s_binary


def apply_color_transform(img, debug=0):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    channel1 = hls[:, :, 2]
    channel2 = img[:, :, 2]
    channel3 = hsv[:, :, 2]
    channel4 = hls[:, :, 1]


    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    channel3 = hls[:, :, 2]
    channel3 = clahe.apply(channel3)
    channel3_filtered = np.zeros_like(channel3)
    channel3_filtered[channel1 > 180] = 1
    channel3_mask = compute_binary(
        channel3_filtered, ksize=19,
        thresholds=dict(sobelx=(200, 255), sobely=(0, 10), magnitude=(50, 255), directional=(0.7, 1.3))
    )

    channel1_mask = compute_binary(
        channel1, ksize=19,
        thresholds=dict(sobelx=(200, 255), sobely=(0, 10), magnitude=(50, 255), directional=(0.7, 1.3))
    )
    channel2_mask = compute_binary(
        channel2, ksize=19,
        thresholds=dict(sobelx=(200, 255), sobely=(0, 10), magnitude=(50, 255), directional=(0.7, 1.3))
    )
    channel4_mask = compute_binary(
        channel4, ksize=19,
        thresholds=dict(sobelx=(150, 255), sobely=(0, 10), magnitude=(200, 255), directional=(0.7, 1.3))
    )

    # Stack each channel
    color_binary = np.dstack((channel1_mask, channel4_mask, np.zeros_like(channel4_mask))) * 255

    # ((channel3 < 70) & (channel3 > 30) & (channel1 > 100)) |
    binary = np.zeros_like(channel1)
    binary[(channel3_mask == 1)] = 1
    binary[((channel4_mask == 1) & (channel1 > 60))] = 1
    binary[(channel1_mask == 1) & (channel1 > 60)] = 1
    binary[(channel2_mask == 1) & (channel4 > 200)] = 1

    if debug >= 2:
        plt.imsave('windows.png', binary)
        import pdb; pdb.set_trace()

    return color_binary, binary


def transform_perspective(image, src, dst):
    image = np.copy(image)
    shape = image.shape
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, shape[::-1], flags=cv2.INTER_LINEAR)
    return warped


def identify_lines(warped, peaks, nwindows=7, debug=0):
    window_height = warped.shape[0] // nwindows
    window_width = 200

    left_x_points = []
    left_y_points = []
    right_x_points = []
    right_y_points = []

    for window in range(nwindows):

        left_window_left = (left_x_points[-1] if left_x_points else peaks[0]) - window_width // 2
        left_window_right = left_window_left + window_width

        right_window_left = (right_x_points[-1] if right_x_points else peaks[1]) - window_width // 2
        right_window_right = right_window_left + window_width

        window_top = warped.shape[0] - (window_height * (window+1))
        window_bottom = window_top + window_height

        left_window = warped[window_top:window_bottom, left_window_left:left_window_right]
        right_window = warped[window_top:window_bottom, right_window_left:right_window_right]

        last_left_value_x = left_x_points[-1] if left_x_points else peaks[0]
        last_left_value_y = window_top + (window_bottom - window_top) // 2
        last_right_value_x = right_x_points[-1] if right_x_points else peaks[1]
        last_right_value_y = window_top + (window_bottom - window_top) // 2

        left_value_y = window_top + (window_bottom - window_top) // 2
        right_value_y = window_top + (window_bottom - window_top) // 2

        def get_value(nonzero, ref):
            return ref + int(np.mean(nonzero))

        left_nonzero_x = left_window.nonzero()[1]    # + left_window_left
        right_nonzero_x = right_window.nonzero()[1]  # + right_window_left
        left_nonzero_y = left_window.nonzero()[0]    # + window_top
        right_nonzero_y = right_window.nonzero()[0]  # + window_top

        left_magnitude = len(left_nonzero_x)
        right_magnitude = len(right_nonzero_x)
        magnitude_threshold = max(0.1 * max(left_magnitude, right_magnitude), 50)

        left_value_x = None
        right_value_x = None

        if left_magnitude >= magnitude_threshold and right_magnitude < magnitude_threshold:
            left_value_x = get_value(left_nonzero_x, left_window_left)
        elif right_magnitude >= magnitude_threshold and left_magnitude < magnitude_threshold:
            right_value_x = get_value(right_nonzero_x, right_window_left)
        elif left_magnitude <= magnitude_threshold and right_magnitude <= magnitude_threshold:
            pass
            left_value_x = int(last_left_value_x + compute_avg_diff(left_x_points))
            right_value_x = int(last_right_value_x + compute_avg_diff(right_x_points))
        else:
            left_value_x = get_value(left_nonzero_x, left_window_left)
            left_value_y = get_value(left_nonzero_y, window_top)
            right_value_x = get_value(right_nonzero_x, right_window_left)
            right_value_y = get_value(right_nonzero_y, window_top)

        if left_value_x is not None:
            for idx in range(4):
                left_x_points.append(int(left_value_x))
                left_y_points.append(int(left_value_y-1))
                left_x_points.append(int(left_value_x))
                left_y_points.append(int(left_value_y+1))
        if right_value_x is not None:
            for idx in range(4):
                right_x_points.append(int(right_value_x))
                right_y_points.append(int(right_value_y-idx))
                right_x_points.append(int(right_value_x))
                right_y_points.append(int(right_value_y+idx))

        if debug:
            debug_image = np.copy(warped) * 255
            debug_image = np.dstack((debug_image, debug_image, debug_image))
            if left_value_x is not None:
                cv2.drawMarker(debug_image, (left_value_x, left_value_y), color=(0,0,255), thickness=2)
            if right_value_x is not None:
                cv2.drawMarker(debug_image, (right_value_x, right_value_y), color=(255,0,0), thickness=2)
            cv2.rectangle(
                debug_image,
                (left_window_left, window_top),
                (left_window_right, window_bottom),
                (0, 255, 0),
                2
            )
            cv2.rectangle(
                debug_image,
                (right_window_left, window_top),
                (right_window_right, window_bottom),
                (0, 255, 0),
                2
            )

            plt.imsave('windows.png', debug_image)
            if debug >= 2:
                import pdb; pdb.set_trace()

    return left_x_points, left_y_points, right_x_points, right_y_points


def pipeline(init, image, debug=0):
    init['frameno'] += 1
    [undistorted_image] = undistort([image])
    color_binary_image, binary_image = apply_color_transform(undistorted_image)
    cropped_image = crop(np.copy(binary_image), 440, debug=debug)
    warped = transform_perspective(cropped_image, init['src'], init['dst'])

    # eroded = cv2.erode(warped, np.ones((3, 3)))
    warped = cv2.dilate(warped, np.ones((7, 3)))

    if not init['peaks'][0] or not init['peaks'][1]:
        init['peaks'] = find_peaks(warped, init['lane_width'])

    left_x_points, left_y_points, right_x_points, right_y_points = identify_lines(
            warped, init['peaks'], nwindows=7, debug=debug)

    if len(left_x_points) and len(right_x_points):
        curr_lane_width = right_x_points[0] - left_x_points[0]
        init['lane_width'] = int(init['lane_width'] * 0.1 + curr_lane_width * 0.9)
        init['peaks'] = [int(left_x_points[0]), int(right_x_points[0])]

    init['last_fit'][0] = get_fit(left_x_points, left_y_points, init.get('last_fit', [None, None])[0])
    init['last_fit'][1] = get_fit(right_x_points, right_y_points, init.get('last_fit', [None, None])[1])

    result = annotate_image(init, warped, undistorted_image, left_x_points, right_x_points, debug=debug)
    return result


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    subparsers = parser.add_subparsers()


    parser_calibrate = subparsers.add_parser('calibrate')
    parser_undistort = subparsers.add_parser('undistort')
    parser_pipeline = subparsers.add_parser('pipeline')
    parser_video = subparsers.add_parser('video')

    parser_calibrate.add_argument('--action', type=str, default='calibrate')
    parser_undistort.add_argument('--action', type=str, default='undistort')
    parser_pipeline.add_argument('--action', type=str, default='pipeline')
    parser_video.add_argument('--action', type=str, default='video')

    parser_pipeline.add_argument('image', type=str)
    parser_pipeline.add_argument('--debug', dest='debug', type=int, default=0)
    parser_video.add_argument('video_in', type=str)
    parser_video.add_argument('video_out', type=str)
    parser_video.add_argument('--debug', dest='debug', type=int, default=0)
    parser_video.add_argument('--start', type=float, default=-1)
    parser_video.add_argument('--end', type=float, default=-1)

    arguments = parser.parse_args()

    init = {}
    init.setdefault('frameno', -1)
    init.setdefault('lane_width', 410)
    init.setdefault('peaks', [None, None, None])
    init.setdefault('last_fit', [None, None])
    init.setdefault('bottom_diff', None)
    init.setdefault('top_diff', None)
    init.setdefault('tr', 10)
    init.setdefault('window', defaultdict(dict))
    src_top_xoffset = 21
    init['src'] = np.float32([
        [ 880.,  700.],
        [ 400.,  700.],
        [ 590. + src_top_xoffset,  455.],
        [ 690. - src_top_xoffset,  455.]
    ])
    init['dst'] = np.float32([
        [ 770.,  720.],
        [ 530.,  720.],
        [ 530.,  0.],
        [ 770.,  0.]
    ])

    if arguments.action == 'calibrate':
        calibrate([cv2.imread(it) for it in glob.glob('camera_cal/*')])
    elif arguments.action == 'undistort':
        images = [cv2.imread(it) for it in glob.glob('camera_cal/*')]
        corrected_images = undistort(images)
        for pair in zip(images, corrected_images):
            save_figure(pair)
    elif arguments.action == 'pipeline':
        image = cv2.imread(arguments.image)
        pipeline(init, image, debug=arguments.debug)
    elif arguments.action == 'video':
        from moviepy.editor import VideoFileClip
        video_in = VideoFileClip(arguments.video_in)
        if arguments.start >= 0 and arguments.end > arguments.start:
            video_in = video_in.subclip(arguments.start, arguments.end)
        video_out = video_in.fl_image(partial(pipeline, init, debug=arguments.debug))
        video_out.write_videofile(arguments.video_out, audio=False)

if __name__ == "__main__":
    main()
