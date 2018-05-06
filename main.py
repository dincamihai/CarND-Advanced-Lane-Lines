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


def save_image(images, layout=(1, 2)):
    f, axs = plt.subplots(*layout, figsize=(24, 9))
    f.tight_layout()
    for idx, (ax, image) in enumerate(zip(axs, images)):
        ax.imshow(image)
        ax.set_title('Image %s' % idx, fontsize=50)
    _uid = uuid.uuid4()
    f.savefig('tmp/orig-%s-%s.png' % (_uid, datetime.datetime.now().strftime("%d-%H-%M-%S")))


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


def undistort(orig_images, mtx, dist):
    images = []

    for _img in orig_images:
        images.append(cv2.undistort(_img, mtx, dist, None, mtx))

    return images


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--action', choices=['calibrate'], required=True)
    arguments = parser.parse_args()

    if arguments.action == 'calibrate':

        mtx, dist = None, None

        with open('cal_matrix.p', 'rb') as _f:
            mtx = pickle.load(_f)
        with open('cal_dist.p', 'rb') as _f:
            dist = pickle.load(_f)

        images = [cv2.imread(it) for it in glob.glob('camera_cal/*')]

        if mtx is None or dist is None:
            cal_images, mxt, dist = calibrate(images)

        corrected_images = undistort(images, mtx, dist)

        for pair in zip(images, corrected_images):
            save_image(pair)


if __name__ == "__main__":
    main()
