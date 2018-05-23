import cv2
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


def compute_avg_diff(points):
    if len(points) > 1:
        return np.mean([(it2-it1) for (it1,it2) in zip(points[0::2], points[1::2])])
    else:
        return 0


def find_peaks(image, lane_width):
    histogram = np.sum(image[(image.shape[0]//2):,:], axis=0)
    middle = histogram.shape[0] // 2
    rel_right = np.argmax(histogram[middle:])
    right = (middle + rel_right) if rel_right else 0
    left = np.argmax(histogram[:middle])
    if left and not right:
        right = left + lane_width
    elif right and not left:
        left = right - lane_width
    return [left, right]


def get_fit(x_points, y_points, last_fit):
    curve_attenuation_factor = 0.1
    raw_fit = np.polyfit(y_points, x_points, 2) if len(x_points) else last_fit
    last_fit = last_fit if last_fit is not None else raw_fit
    att_fit = curve_attenuation_factor * raw_fit + (1-curve_attenuation_factor) * last_fit
    return att_fit


def compute_deviation(image_center, left_fitx, right_fitx, xm_per_pix):
    lane_center = (right_fitx[0] - left_fitx[0]) // 2 + left_fitx[0]
    lane_center_deviation = (lane_center - image_center) * xm_per_pix
    deviation_side = 'left' if lane_center < image_center else 'right'
    deviation_side = '-' if not lane_center_deviation else deviation_side
    return lane_center_deviation, deviation_side


def compute_curvature(ploty, xm_per_pix, ym_per_pix, fitx):
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)
    y_eval = np.max(ploty)

    # Calculate the new radii of curvature
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad


def annotate_image(init, warped, undistorted_image, left_x_points, right_x_points, debug=0):
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = init['last_fit'][0][0]*ploty**2 + init['last_fit'][0][1]*ploty + init['last_fit'][0][2]
    right_fitx = init['last_fit'][1][0]*ploty**2 + init['last_fit'][1][1]*ploty + init['last_fit'][1][2]


    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

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

    curvature_func = partial(compute_curvature, ploty, xm_per_pix, ym_per_pix)

    if init['frameno'] % 30 == 0:
        curverad = curvature_func(left_fitx) if len(left_x_points) > len(right_x_points) else curvature_func(right_fitx)
        init['curverad'] = curverad

    image_center = result.shape[1] // 2
    lane_center_deviation, deviation_side = compute_deviation(image_center, left_fitx, right_fitx, xm_per_pix)

    cv2.putText(
        result,
        'radius: %.2f m deviation: %.2f m %s' %(
        init.get('curverad', None),
        abs(lane_center_deviation), deviation_side),
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA
    )
    return result, ploty, left_fitx, right_fitx
