# -*- coding: utf-8 -*-

#!/usr/bin/env python

import cv2
import glob
import pickle
import argparse
import matplotlib
import datetime
import uuid
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from color import abs_sobel_thresh, mag_thresh, dir_threshold


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


def crop(image):
    image[0:440, :, :] = 0
    image[image.shape[0]-20:, :, :] = 0
    return image


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


def apply_color_transform(img):
    # Convert to HLS color space and separate the V channel
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    channel1 = hls[:, :, 2]
    channel2 = hsv[:, :, 1]
    channel3 = img[:, :, 0]

    channel1_mask = compute_binary(
        channel1, ksize=19,
        thresholds=dict(sobelx=(200, 255), sobely=(0, 10), magnitude=(50, 255), directional=(0.7, 1.3))
    )

    channel2_mask = compute_binary(
        channel2, ksize=19,
        thresholds=dict(sobelx=(200, 255), sobely=(0, 10), magnitude=(50, 255), directional=(0.7, 1.3))
    )

    channel3_mask = compute_binary(
        channel3, ksize=19,
        thresholds=dict(sobelx=(200, 255), sobely=(0, 10), magnitude=(200, 255), directional=(0.7, 1.3))
    )
    # channel3_mask = np.zeros_like(channel3)

    # Stack each channel
    color_binary = np.dstack((channel1_mask, channel2_mask, channel3_mask)) * 255

    binary = np.zeros_like(channel1)
    binary[(channel1_mask == 1) | (channel2_mask == 1) | (channel3_mask == 1)] = 1
    return color_binary, binary


def get_transform_dst(shape, xoffset=0, yoffset=0):
    return np.float32([
        [shape[1]-xoffset, yoffset],
        [shape[1]-xoffset, shape[0]-yoffset],
        [xoffset, shape[0]-yoffset],
        [xoffset, yoffset],
    ])


def transform_perspective(image, src, shape=None, xoffset=0, yoffset=0):
    if shape is None:
        shape = image.shape
    dst = get_transform_dst(shape, xoffset=xoffset, yoffset=yoffset)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, image.shape[::-1], flags=cv2.INTER_LINEAR)
    return warped


def pipeline(image):

    [undistorted_image] = undistort([image])
    cropped_image = crop(undistorted_image)
    color_binary_image, binary_image = apply_color_transform(cropped_image)
    src = np.float32([
        [710, 466],
        [1090, 710],
        [220, 710],
        [570, 466],
    ])
    xoffset = 300
    yoffset = 0
    warped = transform_perspective(binary_image, src, shape=undistorted_image.shape, xoffset=xoffset, yoffset=yoffset)

    f, axs = plt.subplots(2, 3, figsize=(30, 10))
    f.tight_layout()

    axs[0][0].imshow(image)
    axs[0][1].imshow(color_binary_image)
    axs[0][2].imshow(binary_image, cmap='gray')

    axs[1][0].imshow(binary_image, cmap='gray')
    axs[1][1].imshow(warped, cmap='gray')

    dst = get_transform_dst(undistorted_image.shape, xoffset=xoffset, yoffset=yoffset)

    axs[1][0].plot(*src[0], 'o')
    axs[1][0].plot(*src[1], '*')
    axs[1][0].plot(*src[2], 'x')
    axs[1][0].plot(*src[3], '+')

    axs[1][1].plot(*dst[0], 'o')
    axs[1][1].plot(*dst[1], '*')
    axs[1][1].plot(*dst[2], 'x')
    axs[1][1].plot(*dst[3], '+')

    nwindows = 5
    window_height = warped.shape[0] // nwindows

    left_x_points = []
    left_y_points = []
    right_x_points = []
    right_y_points = []

    def find_peaks(histogram):
        middle = histogram.shape[0] // 2
        right = middle + np.argmax(histogram[middle:])
        left = np.argmax(histogram[:middle])
        return left, right

    threshold = 50
    for window in range(nwindows):
        top = warped.shape[0] - (window_height * (window+1))
        bottom = top + window_height
        histogram = np.sum(warped[top:bottom,:], axis=0)
        left_peak, right_peak = find_peaks(histogram)
        print(top)
        print(bottom)
        y_coord = (top+bottom) // 2
        print(y_coord)
        left_peak_value = histogram[left_peak]
        right_peak_value = histogram[right_peak]

        left_factor = 0.3
        right_factor = 0.3
        left_right_distance = 700
        comparison_factor = 5

        last_left_peak = left_peak
        if left_x_points:
            last_left_peak = left_x_points[-1]
        else:
            left_factor = 1

        last_right_peak = right_peak
        if right_x_points:
            last_right_peak = right_x_points[-1]
        else:
            right_factor = 1

        if left_peak_value > (comparison_factor * right_peak_value):
            right_peak = left_peak + left_right_distance
        elif right_peak_value > (comparison_factor * left_peak_value):
            left_peak = right_peak - left_right_distance

        right_peak = right_peak * right_factor + last_right_peak * (1-right_factor)
        left_peak = left_peak * left_factor + last_left_peak * (1-left_factor)

        left_x_points.append(left_peak)
        left_y_points.append(y_coord)
        right_x_points.append(right_peak)
        right_y_points.append(y_coord)
        # axs[1][2].plot(left_peak, histogram[left_peak], 'o', ms=10, color='red')
        # axs[1][2].plot(right_peak, histogram[right_peak], 'o', ms=10, color='blue')

    print(list(zip(left_x_points, left_y_points)))
    print(list(zip(right_x_points, right_y_points)))
    left_fit = np.polyfit(left_y_points, left_x_points, 2)
    right_fit = np.polyfit(right_y_points, right_x_points, 2)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    axs[1][2].imshow(warped, cmap='gray')
    axs[1][2].plot(left_fitx, ploty, color='yellow')
    axs[1][2].plot(right_fitx, ploty, color='yellow')

    axs[1][2].plot(left_x_points, left_y_points, 'o', color='red')
    axs[1][2].plot(right_x_points, right_y_points, 'o', color='red')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    f.savefig('figure.png')
    # save_figure([image, binary_image, warped], fname='figure.png', cmaps=[None, None, 'gray'])


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    subparsers = parser.add_subparsers()


    parser_calibrate = subparsers.add_parser('calibrate')
    parser_undistort = subparsers.add_parser('undistort')
    parser_pipeline = subparsers.add_parser('pipeline')

    parser_calibrate.add_argument('--action', type=str, default='calibrate')
    parser_undistort.add_argument('--action', type=str, default='undistort')
    parser_pipeline.add_argument('--action', type=str, default='pipeline')
    parser_pipeline.add_argument('image', type=str)

    arguments = parser.parse_args()

    if arguments.action == 'calibrate':
        calibrate([cv2.imread(it) for it in glob.glob('camera_cal/*')])
    elif arguments.action == 'undistort':
        images = [cv2.imread(it) for it in glob.glob('camera_cal/*')]
        corrected_images = undistort(images)
        for pair in zip(images, corrected_images):
            save_figure(pair)
    elif arguments.action == 'pipeline':
        image = cv2.imread(arguments.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pipeline(image)


if __name__ == "__main__":
    main()
