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
                [500, crop_top],                           # top left
                [image.shape[1]-500, crop_top],            # top right
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
    cv2.bitwise_not(mask2, mask2)
    out = cv2.bitwise_and(out, mask2)
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


def apply_color_transform(img):
    # Convert to HLS color space and separate the V channel
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    channel1 = hls[:, :, 2]
    channel2 = hsv[:, :, 1]
    channel3 = img[:, :, 0]
    channel4 = hsv[:, :, 2]

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

    channel4_mask = compute_binary(
        channel4, ksize=19,
        thresholds=dict(sobelx=(200, 255), sobely=(0, 10), magnitude=(50, 255), directional=(0.7, 1.3))
    )
    # channel3_mask = np.zeros_like(channel3)

    # Stack each channel
    color_binary = np.dstack((channel1_mask, channel2_mask, channel3_mask)) * 255

    binary = np.zeros_like(channel1)
    binary[(channel2_mask == 1) & (hsv[:, :, 2] > 130) | (channel3_mask == 1) | (channel4_mask == 1) & (channel4 > 160)] = 1

    # plt.imsave('layer.png', binary)
    # import pdb; pdb.set_trace()

    return color_binary, binary


def transform_perspective(image, src, dst):
    image = np.copy(image)
    shape = image.shape
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, shape[::-1], flags=cv2.INTER_LINEAR)

    # clean
    # warped[:, :xoffset-60] = 0
    # warped[:, warped.shape[1]-xoffset+200:] = 0
    return warped


def find_peaks(histogram):
    middle = histogram.shape[0] // 2
    right = middle + np.argmax(histogram[middle:])
    left = np.argmax(histogram[:middle])
    magnitude_factor = 3
    ref = 'left' if histogram[left] > histogram[right] else 'right'
    # if histogram[left] >= magnitude_factor * histogram[right]:
    #     right = left + detected_width or 0
    # elif histogram[right] >= magnitude_factor * histogram[left]:
    #     left = right - detected_width or 0
    # plt.plot(histogram)
    # plt.savefig('windows.png')
    # import pdb; pdb.set_trace()
    return [left, right, ref]


def get_coordinates_conv(warped, init=None, nwindows=7, debug=0):

    # window settings
    window_width = 50
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 50 # How much to slide left and right for searching

    def window_mask(width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output


    def find_window_centroids(image, window_width, window_height, margin):
        left_x_points = []
        left_y_points = []
        right_x_points = []
        right_y_points = []
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(image.shape[0]/2):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2

        r_sum = np.sum(image[int(image.shape[0]/2):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

        if debug == 2:
            debug_image = np.copy(image) * 255
            debug_image = np.dstack((debug_image, debug_image, debug_image))
            cv2.rectangle(
                debug_image,
                (0, int(image.shape[0]/2)),
                (int(image.shape[1]/2), image.shape[0]),
                (0, 255, 0), 2
            )
            cv2.rectangle(
                debug_image,
                (int(image.shape[1]/2), int(image.shape[0]/2)),
                (image.shape[1], image.shape[0]),
                (255, 100, 20), 2
            )
            cv2.drawMarker(debug_image, (int(l_center), image.shape[0]), color=(0,0,255), thickness=2)
            cv2.drawMarker(debug_image, (int(r_center), image.shape[0]), color=(0,0,255), thickness=2)
            plt.imsave('windows.png', debug_image)
            import pdb; pdb.set_trace()

        # Add what we found for the first layer
        center_y = (image.shape[0]-window_height + image.shape[0]) // 2
        left_x_points.append(l_center)
        right_x_points.append(r_center)
        left_y_points.append(center_y)
        right_y_points.append(center_y)

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)

            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            import pdb; pdb.set_trace()
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

            if debug == 2:
                debug_image = np.copy(image) * 255
                debug_image = np.dstack((debug_image, debug_image, debug_image))
                cv2.rectangle(
                    debug_image,
                    (0, int(image.shape[0]-(level+1)*window_height)),
                    (image.shape[1], int(image.shape[0]-level*window_height)),
                    (0, 255, 0), 2
                )
                center_y = (image.shape[0]-(level+1)*window_height + image.shape[0]-level*window_height) // 2
                cv2.drawMarker(debug_image, (int(l_center), center_y), color=(0,0,255), thickness=2)
                cv2.drawMarker(debug_image, (int(r_center), center_y), color=(0,0,255), thickness=2)
                plt.imsave('windows.png', debug_image)
                import pdb; pdb.set_trace()

            left_x_points.append(l_center)
            right_x_points.append(r_center)
            left_y_points.append(center_y)
            right_y_points.append(center_y)

        return left_x_points, left_y_points, right_x_points, right_y_points

    return find_window_centroids(warped, window_width, window_height, margin)


def get_coordinates(warped, init=None, nwindows=7, debug=0):
    window_height = warped.shape[0] // nwindows
    window_width = 200

    left_x_points = []
    left_y_points = []
    right_x_points = []
    right_y_points = []

    for window in range(nwindows):

        left_window_left = (left_x_points[-1] if left_x_points else init['peaks'][0]) - window_width // 2
        left_window_right = left_window_left + window_width

        right_window_left = (right_x_points[-1] if right_x_points else init['peaks'][1]) - window_width // 2
        right_window_right = right_window_left + window_width

        window_top = warped.shape[0] - (window_height * (window+1))
        window_bottom = window_top + window_height

        left_window = warped[window_top:window_bottom, left_window_left:left_window_right]
        right_window = warped[window_top:window_bottom, right_window_left:right_window_right]

        left_nonzero_x = left_window.nonzero()[1]    # + left_window_left
        right_nonzero_x = right_window.nonzero()[1]  # + right_window_left
        left_nonzero_y = left_window.nonzero()[0]    # + window_top
        right_nonzero_y = right_window.nonzero()[0]  # + window_top

        left_x_mean = None
        right_x_mean = None
        left_y_mean = None
        right_y_mean = None
        left_magnitude = len(left_nonzero_x)
        right_magnitude = len(right_nonzero_x)

        if left_magnitude:
            left_x_mean = int(np.mean(left_nonzero_x))
            left_y_mean = int(np.mean(left_nonzero_y))
        if right_magnitude:
            right_x_mean = int(np.mean(right_nonzero_x))
            right_y_mean = int(np.mean(right_nonzero_y))

        last_left_value_x = left_x_points[-1] if left_x_points else init['peaks'][0]
        last_left_value_y = window_top + (window_bottom - window_top) // 2
        last_right_value_x = right_x_points[-1] if right_x_points else init['peaks'][1]
        last_right_value_y = window_top + (window_bottom - window_top) // 2

        if left_magnitude == right_magnitude == 0:
            def compute_avg_diff(points):
                if len(points) > 1:
                    return np.mean([(it2-it1) for (it1,it2) in zip(points[0::2], points[1::2])])
                else:
                    return 0
            left_value_x = int(last_left_value_x + compute_avg_diff(left_x_points))
            right_value_x = int(last_right_value_x + compute_avg_diff(right_x_points))
            left_value_y = window_top + (window_bottom - window_top) // 2
            right_value_y = left_value_y
            left_x_points.append(int(left_value_x))
            left_y_points.append(int(left_value_y))
            right_x_points.append(int(right_value_x))
            right_y_points.append(int(right_value_y))
        else:
            ref = None
            if left_x_mean and (not right_x_mean or right_magnitude < 50):
                ref = 'left'
                right_x_mean = left_x_mean
                right_y_mean = left_y_mean
            elif right_x_mean and (not left_x_mean or left_magnitude < 50):
                ref = 'right'
                left_x_mean = right_x_mean
                left_y_mean = right_y_mean
            elif not (left_x_mean and right_x_mean):
                left_x_mean = last_left_value_x - left_window_left
                left_y_mean = last_left_value_y - window_top
                right_x_mean = last_right_value_x - right_window_left
                right_y_mean = last_right_value_y - window_top

            if ref == 'left':
                left_value_x = left_x_mean + left_window_left
                # right_value_x = left_value_x + init['lane_width']
                right_value_x = right_window_left + left_x_mean
                left_value_y = left_y_mean + window_top
                right_value_y = left_value_y
            elif ref == 'right':
                right_value_x = right_x_mean + right_window_left
                # left_value_x = right_value_x - init['lane_width']
                left_value_x = left_window_left + right_x_mean
                right_value_y = right_y_mean + window_top
                left_value_y = right_value_y
            else:
                left_value_x = left_window_left + left_x_mean
                right_value_x = right_window_left + right_x_mean
                left_value_y = left_y_mean + window_top
                right_value_y = right_y_mean + window_top

            left_x_points.append(int(left_value_x))
            left_y_points.append(int(left_value_y))
            right_x_points.append(int(right_value_x))
            right_y_points.append(int(right_value_y))

        if debug:
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


def pipeline(init, image, debug=0):
    [undistorted_image] = undistort([image])
    color_binary_image, binary_image = apply_color_transform(undistorted_image)
    cropped_image = crop(np.copy(binary_image), 440, debug=debug)
    src_top_xoffset = 80
    xoffset = 250
    shape = cropped_image.shape
    initial_bottom_diff = None
    image_center = image.shape[1] // 2
    lane_center = image_center
    lane_width = 300
    crop_top = 510
    while True and init['frameno'] == 0:
        init['src'] = np.float32([
            [image_center+xoffset, shape[0]],  # bottom right
            [image_center-xoffset, shape[0]],           # bottom left
            [image_center-src_top_xoffset, crop_top],       # top left
            [image_center+src_top_xoffset, crop_top],       # top right
        ])
        init['dst'] = np.float32([
            [image_center+lane_width//2, shape[0]],  # bottom right
            [image_center-lane_width//2, shape[0]],           # bottom left
            [image_center-lane_width//2, 0],                  # top left
            [image_center+lane_width//2, 0],         # top right
        ])
        warped = transform_perspective(cropped_image, init['src'], init['dst'])
        bottom_hist = np.sum(warped[(warped.shape[0]//2):,:], axis=0)
        bottom_left, bottom_right, bottom_ref = find_peaks(bottom_hist)
        top_hist = np.sum(warped[:(warped.shape[0]//2),:], axis=0)
        top_left, top_right, top_ref = find_peaks(top_hist)
        top_diff = top_right - top_left
        bottom_diff = bottom_right - bottom_left
        if initial_bottom_diff is None:
            initial_bottom_diff = bottom_diff
        bottom_lane_center = (bottom_right - bottom_left) // 2 + bottom_left
        bottom_lane_center_deviation = image_center - bottom_lane_center
        top_lane_center = (top_right - top_left) // 2 + top_left
        top_lane_center_deviation = image_center - top_lane_center
        print("Bottom lane center: %s" %  bottom_lane_center)
        print("Top lane center: %s" % top_lane_center)
        print("Crop top: %s" % crop_top)
        warped = transform_perspective(cropped_image, init['src'], init['dst'])
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
            print(top_diff)
            print(bottom_diff)
            print(src_top_xoffset)
            import pdb; pdb.set_trace()
        if top_diff < initial_bottom_diff:
            src_top_xoffset -= 10
        else:
            break

    warped = transform_perspective(cropped_image, init['src'], init['dst'])

    # clean
    # warped[:, :300] = 0
    # warped[:, 500:850] = 0
    # warped[:, 1100:] = 0

    if init['lane_width'] is None:
        bottom_hist = np.sum(warped[(warped.shape[0]//2):,:], axis=0)
        init['peaks'] = find_peaks(bottom_hist)
        top_hist = np.sum(warped[:(warped.shape[0]//2),:], axis=0)
        top_left, top_right, top_ref = find_peaks(top_hist)
        init['lane_width'] = ((init['peaks'][1] - init['peaks'][0]) + (top_right - top_left)) // 2

    nwindows = 7

    left_x_points, left_y_points, right_x_points, right_y_points = get_coordinates(
            warped, init=init, nwindows=nwindows, debug=debug)

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

    if init['frameno'] % 1 == 0:
        # Calculate the new radii of curvature
        init['left_curverad'] = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        init['right_curverad'] = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    init['frameno'] += 1

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    Minv = cv2.getPerspectiveTransform(init['dst'], init['src'])
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_image.shape[1], undistorted_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)

    lane_center = (right_x_points[0] - left_x_points[0]) // 2 + left_x_points[0]
    image_center = image.shape[1] // 2
    lane_center_deviation = image_center - lane_center
    cv2.putText(
        result,
        'left/right radius: %.2f/%.2f m deviation: %.2f' %(init.get('left_curverad', '-'), init.get('right_curverad', '-'), lane_center_deviation),
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA)


    if debug:
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
    parser_video.add_argument('--start', type=float)
    parser_video.add_argument('--end', type=float)

    arguments = parser.parse_args()

    init = {}
    init.setdefault('frameno', 0)
    init.setdefault('lane_width', None)
    init.setdefault('peaks', [None, None, None])
    init.setdefault('last_fit', [None, None])

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
        pipeline(init, image, debug=arguments.debug)
    elif arguments.action == 'video':
        from moviepy.editor import VideoFileClip
        video_in = VideoFileClip(arguments.video_in)
        if arguments.start and arguments.end:
            video_in = video_in.subclip(arguments.start, arguments.end)
        video_out = video_in.fl_image(partial(pipeline, init, debug=arguments.debug))
        video_out.write_videofile(arguments.video_out, audio=False)

if __name__ == "__main__":
    main()
