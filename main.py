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
                [640, crop_top+50],                           # top left
                [image.shape[1]-640, crop_top+50],            # top right
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


def find_peaks(histogram):
    middle = histogram.shape[0] // 2
    rel_right = np.argmax(histogram[middle:])
    right = (middle + rel_right) if rel_right else 0
    left = np.argmax(histogram[:middle])
    magnitude_left = histogram[left]
    magnitude_right = histogram[right]
    default_lane_width = 520
    if magnitude_left < 50:
        left = right - default_lane_width
    if magnitude_right < 50:
        right = left + default_lane_width
    ref = 'left' if magnitude_left > magnitude_right else 'right'
    return [left, right, ref]


def identify_lines(warped, init=None, peaks=None, nwindows=7, debug=0, debug_frame=0):
    window_height = warped.shape[0] // nwindows
    window_width = 200

    left_x_points = []
    left_y_points = []
    right_x_points = []
    right_y_points = []

    def compute_avg_diff(points):
        if len(points) > 1:
            return np.mean([(it2-it1) for (it1,it2) in zip(points[0::2], points[1::2])])
        else:
            return 0

    for window in range(nwindows):

        left_window_left = (left_x_points[-1] if left_x_points else peaks[0]) - window_width // 2
        left_window_right = left_window_left + window_width

        right_window_left = (right_x_points[-1] if right_x_points else peaks[1]) - window_width // 2
        right_window_right = right_window_left + window_width

        window_top = warped.shape[0] - (window_height * (window+1))
        window_bottom = window_top + window_height

        left_window = warped[window_top:window_bottom, left_window_left:left_window_right]
        right_window = warped[window_top:window_bottom, right_window_left:right_window_right]

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
        magnitude_threshold = max(0.05 * max(left_magnitude, right_magnitude), 50)

        last_left_value_x = left_x_points[-1] if left_x_points else peaks[0]
        last_left_value_y = window_top + (window_bottom - window_top) // 2
        last_right_value_x = right_x_points[-1] if right_x_points else peaks[1]
        last_right_value_y = window_top + (window_bottom - window_top) // 2

        if left_magnitude >= magnitude_threshold and right_magnitude < magnitude_threshold:
            left_value_x = get_value(left_nonzero_x, left_window_left)
            right_value_x = int(last_right_value_x + compute_avg_diff(right_x_points))
            right_value_x = init['window'].get(window, {}).get('right', [None, None])[0] or right_value_x
            init['window'][window]['left'] = (left_value_x, left_value_y)
        elif right_magnitude >= magnitude_threshold and left_magnitude < magnitude_threshold:
            right_value_x = get_value(right_nonzero_x, right_window_left)
            left_value_x = int(last_left_value_x + compute_avg_diff(left_x_points))
            left_value_x = init['window'].get(window, {}).get('left', [None, None])[0] or left_value_x
            init['window'][window]['right'] = (right_value_x, right_value_y)
        elif left_magnitude <= magnitude_threshold and right_magnitude <= magnitude_threshold:
            left_value_x = int(last_left_value_x + compute_avg_diff(left_x_points))
            right_value_x = int(last_right_value_x + compute_avg_diff(right_x_points))
        else:
            left_value_x = get_value(left_nonzero_x, left_window_left)
            left_value_y = get_value(left_nonzero_y, window_top)
            right_value_x = get_value(right_nonzero_x, right_window_left)
            right_value_y = get_value(right_nonzero_y, window_top)
            init['window'][window]['left'] = (left_value_x, left_value_y)
            init['window'][window]['right'] = (right_value_x, right_value_y)

        left_x_points.append(int(left_value_x))
        left_y_points.append(int(left_value_y))
        right_x_points.append(int(right_value_x))
        right_y_points.append(int(right_value_y))

        if debug and init['frameno'] >= debug_frame:
            debug_image = np.copy(warped) * 255
            debug_image = np.dstack((debug_image, debug_image, debug_image))
            cv2.drawMarker(debug_image, (left_value_x, left_value_y), color=(0,0,255), thickness=2)
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


def pipeline(init, image, debug=0, debug_frame=0):
    init['frameno'] += 1
    [undistorted_image] = undistort([image])
    color_binary_image, binary_image = apply_color_transform(undistorted_image)
    cropped_image = crop(np.copy(binary_image), 440, debug=debug)
    src_top_xoffset = 80
    xoffset = 150
    shape = cropped_image.shape
    initial_bottom_diff = None
    image_center = image.shape[1] // 2
    lane_center = image_center
    crop_top = 510
    while True and not init['dst_found']:
        init['src'] = np.float32([
            [image_center+xoffset, shape[0]],  # bottom right
            [image_center-xoffset, shape[0]],           # bottom left
            [image_center-src_top_xoffset, crop_top],       # top left
            [image_center+src_top_xoffset, crop_top],       # top right
        ])
        init['dst'] = np.float32([
            [image_center+150, shape[0]],  # bottom right
            [image_center-150, shape[0]],           # bottom left
            [image_center-150, 0],                  # top left
            [image_center+150, 0],         # top right
        ])
        warped = transform_perspective(cropped_image, init['src'], init['dst'])

        bottom_hist = np.sum(warped[(3*(warped.shape[0]//4)):warped.shape[0],:], axis=0)
        bottom_left, bottom_right, bottom_ref = find_peaks(bottom_hist)

        if not bottom_left or not bottom_right:
            break

        def find_trasfrom_points(init, warped, bottom_left, bottom_right):
            left_x_points, left_y_points, right_x_points, right_y_points = identify_lines(
                warped, init=init, peaks=(bottom_left, bottom_right), nwindows=7)
            return [
                [left_x_points[0], left_y_points[0]],
                [right_x_points[0], right_y_points[0]],
                [left_x_points[-1], left_y_points[-1]],
                [right_x_points[-1], right_y_points[-1]]
            ]

        bottom_left, bottom_right, top_left, top_right = find_trasfrom_points(
            init, warped, bottom_left, bottom_right)
        init['bottom_diff'] = bottom_right[0] - bottom_left[0]
        init['top_diff'] = top_right[0] - top_left[0]
        init['peaks'] = [bottom_left, bottom_right, bottom_ref]

        init['src1'] = np.float32([bottom_right, bottom_left, top_left, top_right])

        if initial_bottom_diff is None:
            initial_bottom_diff = init['bottom_diff']

        warped = transform_perspective(warped, init['src1'], init['dst'])
        if debug == 2:
            debug_image = np.copy(warped) * 255
            debug_image = np.dstack((debug_image, debug_image, debug_image))

            cv2.drawMarker(debug_image, tuple(init['src'][0]), color=(0, 255, 255), thickness=5)
            cv2.drawMarker(debug_image, tuple(init['src'][1]), color=(255, 255, 0), thickness=5)
            cv2.drawMarker(debug_image, tuple(init['src'][2]), color=(255, 0, 255), thickness=5)
            cv2.drawMarker(debug_image, tuple(init['src'][3]), color=(0, 255, 0), thickness=5)

            cv2.drawMarker(debug_image, tuple(init['dst'][0]), color=(0, 255, 255), thickness=5)
            cv2.drawMarker(debug_image, tuple(init['dst'][1]), color=(255, 255, 0), thickness=5)
            cv2.drawMarker(debug_image, tuple(init['dst'][2]), color=(255, 0, 255), thickness=5)
            cv2.drawMarker(debug_image, tuple(init['dst'][3]), color=(0, 255, 0), thickness=5)
            plt.imsave('windows.png', debug_image)
            print("Top difference: %s" % init['top_diff'])
            print("Bottom difference: %s" % init['bottom_diff'])
            print("Top offset: %s" % src_top_xoffset)
            import pdb; pdb.set_trace()

        init['dst_found'] = True

    if not init['dst_found']:
        return image
    warped = transform_perspective(cropped_image, init['src'], init['dst'])
    warped = transform_perspective(warped, init['src1'], init['dst'])

    eroded = cv2.erode(warped, np.ones((3, 3)))
    warped = cv2.dilate(eroded, np.ones((1, 7)))

    # clean
    # warped[:, :300] = 0
    # warped[:, 400:700] = 0
    # warped[:, 900:] = 0

    left_x_points, left_y_points, right_x_points, right_y_points = identify_lines(
            warped, init=init, peaks=[500, 940], nwindows=7, debug=debug, debug_frame=debug_frame)

    init['peaks'][0], init['peaks'][1] = int(left_x_points[0]), int(right_x_points[0])

    # print(list(zip(left_x_points, left_y_points)))
    # print(list(zip(right_x_points, right_y_points)))
    left_fit = np.polyfit(left_y_points, left_x_points, 2)
    right_fit = np.polyfit(right_y_points, right_x_points, 2)
    print("{0:.4f} {0:.4f} {0:.4f}".format(*(right_fit - left_fit)))

    curve_attenuation_factor = 0.1
    if init['last_fit'][0] is not None:
        left_fit = curve_attenuation_factor * left_fit + (1-curve_attenuation_factor) * init['last_fit'][0]
    if init['last_fit'][1] is not None:
        right_fit = curve_attenuation_factor * right_fit + (1-curve_attenuation_factor) * init['last_fit'][1]

    init['last_fit'][0], init['last_fit'][1] = left_fit, right_fit

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

    if init['frameno'] % 30 == 0:
        # Calculate the new radii of curvature
        init['left_curverad'] = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        init['right_curverad'] = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    Minv1 = cv2.getPerspectiveTransform(init['dst'], init['src1'])
    Minv = cv2.getPerspectiveTransform(init['dst'], init['src'])
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv1, (undistorted_image.shape[1], undistorted_image.shape[0]))
    newwarp = cv2.warpPerspective(newwarp, Minv, (undistorted_image.shape[1], undistorted_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)

    lane_center = (right_x_points[0] - left_x_points[0]) // 2 + left_x_points[0]
    image_center = image.shape[1] // 2
    lane_center_deviation = image_center - lane_center
    cv2.putText(
        result,
        'left/right radius: %.2f/%.2f m deviation: %.2f' %(init.get('left_curverad', '-'), init.get('right_curverad', '-'), lane_center_deviation),
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA)


    if debug and init['frameno'] >= debug_frame:
        f, axs = plt.subplots(2, 3, figsize=(30, 10))
        f.tight_layout()

        axs[0][0].imshow(image)
        axs[0][1].imshow(color_binary_image)
        axs[0][2].imshow(binary_image, cmap='gray')

        axs[1][0].imshow(cropped_image, cmap='gray')
        axs[1][1].imshow(warped, cmap='gray')

        axs[1][0].plot(*init['src'][0], 'o')
        axs[1][0].plot(*init['src'][1], '*')
        axs[1][0].plot(*init['src'][2], 'x')
        axs[1][0].plot(*init['src'][3], '+')

        axs[1][1].plot(*init['dst'][0], 'o')
        axs[1][1].plot(*init['dst'][1], '*')
        axs[1][1].plot(*init['dst'][2], 'x')
        axs[1][1].plot(*init['dst'][3], '+')

        # axs[1][2].imshow(warped, cmap='gray')
        axs[1][1].plot(left_fitx, ploty, color='yellow')
        axs[1][1].plot(right_fitx, ploty, color='yellow')

        axs[1][1].plot(left_x_points, left_y_points, 'o', color='red')
        axs[1][1].plot(right_x_points, right_y_points, 'o', color='red')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        axs[1][2].imshow(result)

        f.savefig('figure.png')
        if debug >= 1:
            plt.imsave('frame.png', image)
            import pdb; pdb.set_trace()

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
    parser_video.add_argument('--debug-frame', dest='debug_frame', type=int, default=0)
    parser_video.add_argument('--start', type=float, default=-1)
    parser_video.add_argument('--end', type=float, default=-1)

    arguments = parser.parse_args()

    init = {}
    init.setdefault('frameno', -1)
    init.setdefault('peaks', [None, None, None])
    init.setdefault('last_fit', [None, None])
    init.setdefault('bottom_diff', None)
    init.setdefault('top_diff', None)
    init.setdefault('tr', 10)
    init.setdefault('window', defaultdict(dict))
    init.setdefault('src', None)
    init.setdefault('dst', None)
    init.setdefault('dst_found', False)
    init.setdefault('left_curverad', None)
    init.setdefault('right_curverad', None)

    if arguments.action == 'calibrate':
        calibrate([cv2.imread(it) for it in glob.glob('camera_cal/*')])
    elif arguments.action == 'undistort':
        images = [cv2.imread(it) for it in glob.glob('camera_cal/*')]
        corrected_images = undistort(images)
        for pair in zip(images, corrected_images):
            save_figure(pair)
    elif arguments.action == 'pipeline':
        image = cv2.imread(arguments.image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        init['src'] = np.float32([
            [ 790.,  720.],
            [ 490.,  720.],
            [ 591.,  510.],
            [ 689.,  510.]
        ])
        init['dst'] = np.float32([
            [ 730.,  720.],
            [ 550.,  720.],
            [ 550.,    0.],
            [ 730.,    0.]
        ])
        init['dst_found'] = True
        pipeline(init, image, debug=arguments.debug)
    elif arguments.action == 'video':
        from moviepy.editor import VideoFileClip
        video_in = VideoFileClip(arguments.video_in)
        if arguments.start >= 0 and arguments.end > arguments.start:
            video_in = video_in.subclip(arguments.start, arguments.end)
        video_out = video_in.fl_image(partial(pipeline, init, debug=arguments.debug, debug_frame=arguments.debug_frame))
        video_out.write_videofile(arguments.video_out, audio=False)

if __name__ == "__main__":
    main()
