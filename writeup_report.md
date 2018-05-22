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

[image2]: ./test_images/test1.jpg "Original Image"
[image2-undist]: ./writeup-images/test1-undist.png "Undistorted Image"
[image2-binary]: ./writeup-images/test1-binary.png "Binary Undistorted Image"
[image2-warped]: ./writeup-images/test1-warped.png "Binary Perspective Transformed Image"
[orig-test2]: ./test_images/straight_lines1.jpg "Original Test #2 Image"
[perspective-test2]: ./writeup-images/perspective-test2.png "Binary Perspective Transformed Test Image #2"
[plotted-result]: ./writeup-images/result.png "Plotted result"
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

I have used a combination of directional gradients with magnitude filtering applied on saturation and value channels of the HSV transformation and the red channel from the RGB image.

Example: Original - Transformed

![alt text][image2-undist] ![alt text][image2-binary]

TODO: add link to code here

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Since we are expecting the transformation to contain two paralel markings, I was able to implemented a mechanism to automatically detect the parameters for the perspective transformation.
I first use hardcoded parameters to get an usable transformation and then I use the histogram to identify 4 points in the image.
I then apply the perspective transformation in a loop, adjusting the distance between the top points on each pass.
It stops when the distance between the top points and the bottom points is almost equal (with some margin)
I only implemented this because hardcoded transformation points were not working the same for the challenge videos.

The automatic detection is only used with videos, for the single image processing I am still using hardcoded parameters.
The hardcoded parameters were automatically found using the method above.

```python
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
```

I verified that the parameters work using the straight lines test images provided.

Straight lines test image trasformed
![alt text][orig-test2] ![alt text][perspective-test2]

And the transformation for the test#1 image looks like this:

Example: Binary - Perspective Transformation
![alt text][image2-binary] ![alt text][image2-warped]

TODO: add link to code here

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for identifying the lane-line positions can be found here:

TODO: add link to code here

To identify the lane lines I'm using the histogram of the bottom half of the image to find the starting points and then I'm using the sliding window technique to find the next points.
I am using 7 windows per image per line which finds 7 (x, y) pairs for each left and right lines.
When no pixels are found, I am estimating based on average variation of the previous points found and previous points found in that area.
When a window contains a number of pixels above a relative threshold, I am computing the mean x and y values of the non-zero pixels.
At the same time I am also updating the global dictionary that keeps track of detected pixel for that current window y layer.

For each frame, I cumulate the left and right (x, y) pairs (both detected and estimated).
I then use them to fit a 2nd grade polynomial and to update the lane-width.
I'm also using an attenuation coeficient for the polynomials in order to smooth out big rapid variations.

TODO: add link to code here

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In order to calculate the radius of curvature I've used this formula:

TODO: add formula here

The implementation can be seen here:

TODO: add link to code here

In order to show the curvature in meters, I've used scaling factors to convert from pixels to meters.

TODO: add link to code here

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The implementation can be seen here:

TODO: add link to code here

Example of result image:

![alt text][plotted-result]


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: add link to video

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The automated perspective transform was implemented because hardcoded parameters did not result in paralel lines in the challenge videos.
It is only enabled when processing videos and, depending on the first frames, if the left and right lines are not identifiable, it might skip some frames until it finds the information needed and only then the detected lane area will be projected over the original image.

The current implementation does not work well with sharp curves, especially when one of the lines disappears from the image (eg: hard challenge video)
One way to overcome this would be to identify situations like this based on the hight curvature of the visible line.

Onother problem that the current implementation has is large areas with shadows and bright spots. The lines are not well identified in those cases.
To overcome this, the part that takes care of generating the binary image needs to be tweaked more.
The challenge would be to find a way that works in all conditions. The histogram equalisation might help here: https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

I also tried erosion and dilation in order to remove noise from the binary image but I decided to use `cv2.bilateralFilter` https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
