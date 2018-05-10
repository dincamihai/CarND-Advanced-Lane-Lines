# -*- coding: utf-8 -*-

#!/usr/bin/env python

import cv2
import glob
import pickle
import argparse
import matplotlib
import datetime
import uuid
# matplotlib.use('Agg')
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
    return warped, dst


def find_peaks(histogram):
    middle = histogram.shape[0] // 2
    right = middle + np.argmax(histogram[middle:])
    left = np.argmax(histogram[:middle])
    return left, right


def get_coordinates(warped, initial_coords=None):
    nwindows = 7
    window_height = warped.shape[0] // nwindows
    window_width = 50

    left_x_points = [initial_coords[0]]
    left_y_points = [0]
    right_x_points = [initial_coords[1]]
    right_y_points = [0]

    left_window_left = initial_coords[0] - window_width // 2
    left_window_right = initial_coords[0] + window_width

    right_window_left = initial_coords[1] - window_width // 2
    right_window_right = initial_coords[1] + window_width

    for window in range(nwindows):
        window_top = warped.shape[0] - (window_height * (window+1))
        window_bottom = window_top + window_height

        left_window = warped[window_top:window_bottom, left_window_left:left_window_right]
        right_window = warped[window_top:window_bottom, right_window_left:right_window_right]

        left_nonzero = left_window.nonzero()[1] + left_window_left
        right_nonzero = right_window.nonzero()[1] + right_window_left

        if len(left_nonzero) > 10:
            left_x_points.append(np.mean(left_nonzero))
            left_y_points.append(window_bottom)
        if len(right_nonzero) > 10:
            right_x_points.append(np.mean(right_nonzero))
            right_y_points.append(window_bottom)


        # axs[1][2].plot(left_peak, histogram[left_peak], 'o', ms=10, color='red')
        # axs[1][2].plot(right_peak, histogram[right_peak], 'o', ms=10, color='blue')
    return left_x_points, left_y_points, right_x_points, right_y_points


def pipeline(image, save_image=False):

    [undistorted_image] = undistort([image])
    cropped_image = crop(np.copy(undistorted_image))
    color_binary_image, binary_image = apply_color_transform(cropped_image)
    src = np.float32([
        [710, 466],
        [1090, 710],
        [220, 710],
        [570, 466],
    ])
    xoffset = 300
    yoffset = 0
    warped, dst = transform_perspective(binary_image, src, shape=undistorted_image.shape, xoffset=xoffset, yoffset=yoffset)

    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    left_peak, right_peak = find_peaks(histogram)

    left_x_points, left_y_points, right_x_points, right_y_points = get_coordinates(
        warped, initial_coords=(left_peak, right_peak))

    # print(list(zip(left_x_points, left_y_points)))
    # print(list(zip(right_x_points, right_y_points)))
    left_fit = np.polyfit(left_y_points, left_x_points, 2)
    right_fit = np.polyfit(right_y_points, right_x_points, 2)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # save_figure([image, binary_image, warped], fname='figure.png', cmaps=[None, None, 'gray'])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    y_eval = np.max(ploty)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_image.shape[1], undistorted_image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)

    if save_figure:
        f, axs = plt.subplots(2, 3, figsize=(30, 10))
        f.tight_layout()

        axs[0][0].imshow(image)
        axs[0][1].imshow(color_binary_image)
        axs[0][2].imshow(binary_image, cmap='gray')

        axs[1][0].imshow(binary_image, cmap='gray')
        axs[1][1].imshow(warped, cmap='gray')

        # dst = get_transform_dst(undistorted_image.shape, xoffset=xoffset, yoffset=yoffset)

        axs[1][0].plot(*src[0], 'o')
        axs[1][0].plot(*src[1], '*')
        axs[1][0].plot(*src[2], 'x')
        axs[1][0].plot(*src[3], '+')

        axs[1][1].plot(*dst[0], 'o')
        axs[1][1].plot(*dst[1], '*')
        axs[1][1].plot(*dst[2], 'x')
        axs[1][1].plot(*dst[3], '+')

        # axs[1][2].imshow(warped, cmap='gray')
        axs[1][1].plot(left_fitx, ploty, color='yellow')
        axs[1][1].plot(right_fitx, ploty, color='yellow')

        axs[1][1].plot(left_x_points, left_y_points, 'o', color='red')
        axs[1][1].plot(right_x_points, right_y_points, 'o', color='red')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        axs[1][2].imshow(result)

        f.savefig('figure.png')
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
    parser_video.add_argument('video_in', type=str)
    parser_video.add_argument('video_out', type=str)

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
        pipeline(image, save_image=True)
    elif arguments.action == 'video':
        from moviepy.editor import VideoFileClip
        video_in = VideoFileClip(arguments.video_in)
        video_out = video_in.fl_image(pipeline)
        video_out.write_videofile(arguments.video_out, audio=False)

if __name__ == "__main__":
    main()
