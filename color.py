import numpy as np
import cv2


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate sobel
    # Apply threshold
    sobel_orient = (1, 0)
    if orient == 'y':
        sobel_orient = (0, 1)
    sobel = cv2.Sobel(img, cv2.CV_64F, *sobel_orient)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    mask = np.zeros_like(scaled_sobel)
    mask[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return mask


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate directional gradient
    # Apply threshold
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absx = np.absolute(sobelx)
    absy = np.absolute(sobely)
    direction = np.arctan2(absy, absx)
    mask = np.zeros(direction.shape)
    mask[(direction>=thresh[0]) & (direction<=thresh[1])] = 1
    return mask
