**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image2]: ./test_images/test2.jpg "Original Image"
[image2-undist]: ./writeup-examples/undist.png "Undistorted Image"
[image2-binary]: ./writeup-examples/binary.png "Binary Undistorted Image"
[image2-warped]: ./writeup-examples/warped.png "Binary Perspective Transformed Image"
[orig-straight-lines]: ./test_images/straight_lines1.jpg "Original Test #2 Image"
[perspective-straight-lines]: ./writeup-examples/perspective.png "Binary Perspective Transformed Test Image #1"
[plotted-result]: ./writeup-examples/result.png "Plotted result"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I order to calibrate the camera, I used `cv2.findChessboardCorners` function to extract the chessboard corners from the provided calibration images.
I cumulated all the extracted corners in a list and, together with a generated reference pattern (Z=0), I used `cv2.calibrateCamera` to obtain the calibration matrix and the distortion coefficients.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Distorted - Undistorted
![alt text][image2] ![alt text][image2-undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I have used a combination of directional gradients with magnitude thresholds applied on saturation and value channels of the HLS and HSV transformation and the red channel from the RGB image.
I have also used `cv2.dilate` to augment the pixels areas obtained after color transformation.

Example: Original - Transformed

![alt text][image2-undist] ![alt text][image2-binary]

The code can be viewed here: https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/main.py#L142-L191

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I've noticed that hardcoded perspective transformation points did not work for both project video and the challenge videos so I tried to implement a mecanism to detect the best transformation. This worked to some degree but I decided to simplify things and to use hardcoded points for the submission.

The points I'm using are the following:

```python
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
```
I have obtained the points by using the straight lines test images provided.

Straight lines test image trasformed
![alt text][orig-straight-lines] ![alt text][perspective-straight-lines]

And the transformation for the test#1 image looks like this:

Example: Binary - Perspective Transformation
![alt text][binary] ![alt text][warped]

The code can be found here: https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/main.py#L194-L199

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for identifying the lane-line positions can be found here: https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/main.py#L202-L301

To identify the lane lines I'm using the histogram of the bottom half of the image to find the starting points and then I'm using the sliding window technique to find the next points.
I am using 7 windows per image per line.
When a window contains a number of pixels above a relative threshold, I am computing the mean x and y values of the non-zero pixels and I am adding 8 points (all have the found `x` and `y` in [found_y-4, found_y+4]) https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/main.py#L262-L273
I am doing this so that I don't risk having too few points which would result in a wrong polinomial fit.

For each frame, I cumulate the left and right (x, y) pairs and I compute and store the lane_width.
I then use them to fit a 2nd grade polynomial and to update the lane-width.
I'm also using an attenuation coeficient for the polynomials in order to smooth out variations.

The code can be found here: https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/main.py#L202-L301

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In order to calculate the radius of curvature I've used this formula:

TODO: add formula here

The code implementation can be seen here: https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/utils.py#L43-L50

In order to show the curvature in meters, I've used scaling factors to convert from pixels to meters.
https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/utils.py#L45

The scaling factors can be seen here: https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/utils.py#L59-L61

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The implementation can be seen here: https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/utils.py#L63-L81

Example of result image:

![alt text][plotted-result]


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

https://raw.githubusercontent.com/dincamihai/CarND-Advanced-Lane-Lines/master/processed_video.mp4

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The current implementation does not work well with sharp curves, especially when one of the lines disappears from the image (eg: hard challenge video)
One way to overcome this would be to identify situations like this based on the hight curvature of the visible line.

Onother problem that I had was identifying lines in images with shadows and bright spots. 
To overcome this, I'm using histogram equalization: https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
https://github.com/dincamihai/CarND-Advanced-Lane-Lines/blob/master/main.py#L153-L154

I also looked at `cv2.bilateralFilter` https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter in order to remove noise from the binary image but I decided to not use it for the submision.
