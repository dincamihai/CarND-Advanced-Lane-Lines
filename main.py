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
        thresholds=dict(sobelx=(200, 255), sobely=(0, 10), magnitude=(50, 255), directional=(0.7, 1.3))
    )

    # Stack each channel
    color_binary = np.dstack((channel1_mask, channel2_mask, channel3_mask)) * 255

    binary = np.zeros_like(channel1)
    binary[(channel1_mask == 1) | (channel2_mask == 1) | (channel3_mask == 1)] = 1
    return color_binary, binary


def get_transform_dst(image, xoffset=0, yoffset=0):
    return np.float32([
        [image.shape[1]-xoffset, yoffset],
        [image.shape[1]-xoffset, image.shape[0]-yoffset],
        [xoffset, image.shape[0]-yoffset],
        [xoffset, yoffset],
    ])


def transform_perspective(image, src, xoffset=0, yoffset=0):
    dst = get_transform_dst(image, xoffset=xoffset, yoffset=yoffset)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, image.shape[::-1], flags=cv2.INTER_LINEAR)
    return warped


def pipeline(image):

    [undistorted_image] = undistort([image])
    color_binary_image, binary_image = apply_color_transform(undistorted_image)
    src = np.float32([
        [710, 465],
        [1040, 676],
        [265, 676],
        [570, 465],
    ])
    xoffset = 300
    yoffset =0
    warped = transform_perspective(binary_image, src, xoffset=xoffset, yoffset=yoffset)

    f, axs = plt.subplots(2, 3, figsize=(30, 10))
    f.tight_layout()

    axs[0][0].imshow(image)
    axs[0][1].imshow(color_binary_image)
    axs[0][2].imshow(binary_image, cmap='gray')

    axs[1][1].imshow(binary_image, cmap='gray')
    axs[1][2].imshow(warped, cmap='gray')

    dst = get_transform_dst(binary_image, xoffset=xoffset, yoffset=yoffset)

    axs[1][2].plot(*dst[0], 'o')
    axs[1][2].plot(*dst[1], '*')
    axs[1][2].plot(*dst[2], 'x')
    axs[1][2].plot(*dst[3], '+')

    axs[1][1].plot(*src[0], 'o')
    axs[1][1].plot(*src[1], '*')
    axs[1][1].plot(*src[2], 'x')
    axs[1][1].plot(*src[3], '+')

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
