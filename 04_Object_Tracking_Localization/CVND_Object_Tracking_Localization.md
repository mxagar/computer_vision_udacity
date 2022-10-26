# Udacity Computer Vision Nanodegree: Object Tracking and Localization

These are my personal notes taken while following the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).

The nanodegree is composed of these modules:

1. Introduction to Computer Vision
2. Cloud Computing (Optional)
3. Advanced Computer Vision and Deep Learning
4. Object Tracking and Localization

Each module has a folder with its respective notes.
This folder/file refers to the **fourth** module: **Object Tracking and Localization**.

Note that:

- I have also notes on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) in my repository [deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity); that MOOC is strongly related and has complementary material. I have many hand-written notes in [deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity) related to this repository, too.
- The exercises are commented in the Markdown files and linked to their location; most of the exercises are located in other repositories, originally forked from Udacity and extended/completed by me:
	- [CVND_Exercises](https://github.com/mxagar/CVND_Exercises)
	- [DL_PyTorch](https://github.com/mxagar/DL_PyTorch)
	- [CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises)
	- [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch)


Mikel Sagardia, 2022.  
No guarantees.

## Practical Installation Notes

I basically followed the installation & setup guide from [CVND_Exercises](https://github.com/mxagar/CVND_Exercises), which can be summarized with the following commands:

```bash
# Create new conda environment to be used for the nanodegree
conda create -n cvnd python=3.6
conda activate cvnd
conda install pytorch torchvision -c pytorch
conda install pip
#conda install -c conda-forge jupyterlab

# Go to the folder where the Udacity DL exercises are cloned, after forking the original repo
cd ~/git_repositories/CVND_Exercises
pip install -r requirements.txt

# I had some issues with numpy and torch
pip uninstall numpy
pip uninstall mkl-service
pip install numpy
pip install mkl-service
```

## Overview of Contents

- [Udacity Computer Vision Nanodegree: Object Tracking and Localization](#udacity-computer-vision-nanodegree-object-tracking-and-localization)
  - [Practical Installation Notes](#practical-installation-notes)
  - [Overview of Contents](#overview-of-contents)
  - [1. Introduction to Motion](#1-introduction-to-motion)
    - [1.1 Introduction to Optical Flow](#11-introduction-to-optical-flow)
    - [1.2 Motion Vectors](#12-motion-vectors)
    - [1.3 Brightness Consistency Assumption](#13-brightness-consistency-assumption)
    - [1.3 Optical Flow Notebook](#13-optical-flow-notebook)
  - [2. Robot Localization](#2-robot-localization)
  - [3. Mini-Project: 2D Histogram Filter](#3-mini-project-2d-histogram-filter)
  - [4. Introduction to Kalman Filters](#4-introduction-to-kalman-filters)
  - [5. Representing State and Motion](#5-representing-state-and-motion)
  - [6. Matrices and Transformation of State](#6-matrices-and-transformation-of-state)
  - [7. Simultaneous Localization and Mapping](#7-simultaneous-localization-and-mapping)
  - [8. Vehicle Motion Calculus (Optional)](#8-vehicle-motion-calculus-optional)
  - [9. Project: Landmark Detection & Tracking (SLAM)](#9-project-landmark-detection--tracking-slam)

## 1. Introduction to Motion

The goal of this module is to show pattern recognition techniques over time and space.

We deep dive into field of **Localization**, with the following topics

- Representing motion and tracking objects in a video
- Uncertainty in robotic motion
- A simple localization technique: Histogram Filter
- Motion models and tracking the position of a self-driving car over time

At the end of this module we work on a SLAM project: Simultaneous Localization and Mapping.

### 1.1 Introduction to Optical Flow

One way of capturing motion in a video consists in extracting features in the frames and observing how they change from frame to frame.

**Optical Flow** works that way; it makes two assumptions:

1. Pixel intensities stay consistent between frames.
2. Neighboring pixels have similar motion.

![Optical Flow: Assumptions](./pics/optical_flow_assumptions.jpg)

Salient pixels are tracked, e.g., bright pixels or corners; tracking them provides information on

- *how fast* the object is moving
- and in what *direction*,

so we are able to predict where it will be.

Applications:

- Hand gesture recognition
- Tracking vehicle movement
- Distinguish running vs. walking
- Eye tracking
- etc.

### 1.2 Motion Vectors

Given a salient/tracked pixel `(x, y)`, its motion can be tracked by a field vector `(u,v)`:

- Magnitude: `sqrt(u^2 + v^2)`
- Direction: `phi = atan(v/u)`

![Motion Vectors](./pics/motion_vectors.jpg)

### 1.3 Brightness Consistency Assumption

We want to get the equation that relates the motion of a pixel (in time: `t`) and the change of the image from frame to frame (change in pixel space: `x, y`). Using the assumptions introduced above, we have:

![Optical Flow: Equations](./pics/optical_flow_equations.jpg)

In practice, we can apply optical flow to keypoints detected by a corner detector. Another option is to apply it to every pixel or to a grid of pixels; in that case, we have a field of velocity vectors.

### 1.3 Optical Flow Notebook

Exercise repository: [CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises).

Notebook: `4_1_Optical_Flow / Optical_Flow.ipynb`.

In the notebook, 3 images that contain a moving packman are used to compute the motion vectors of selected keypoints using optical flow. The example doesn't work that well; possible approaches to improve it: decrease displacement, modify the parameters.

```python
import numpy as np
import matplotlib.image as mpimg  # for reading in images
import matplotlib.pyplot as plt
import cv2  # computer vision library
%matplotlib inline

# Read in the image frames
frame_1 = cv2.imread('images/pacman_1.png')
frame_2 = cv2.imread('images/pacman_2.png')
frame_3 = cv2.imread('images/pacman_3.png')

# Convert to RGB
frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)
frame_3 = cv2.cvtColor(frame_3, cv2.COLOR_BGR2RGB)

# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('frame 1')
ax1.imshow(frame_1)
ax2.set_title('frame 2')
ax2.imshow(frame_2)
ax3.set_title('frame 3')
ax3.imshow(frame_3)

# We need to pass keypoints to track to the Optical Flow API
# We use the Shi-Tomasi corner detector, similar to the Harris corner detector
# We can use Harris or ORB instead, too
# Parameters for Shi-Tomasi corner detection
feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.2,
                       minDistance = 5,
                       blockSize = 5 )

# Convert all frames to grayscale
gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)
gray_3 = cv2.cvtColor(frame_3, cv2.COLOR_RGB2GRAY)

# Take first frame and find corner points in it
pts_1 = cv2.goodFeaturesToTrack(gray_1, mask = None, **feature_params)

# Display the detected points
plt.imshow(frame_1)
for p in pts_1:
    # plot x and y detected points
    plt.plot(p[0][0], p[0][1], 'r.', markersize=15)

# Parameters for Lucas-Kanade optical flow
# winSize: size of the search window at each pyramid level
# maxLevel: 0, pyramids are not used (single level), if set to 1, two levels are used, and so on
# criteria: termination criteria of the iterative search algorithm
lk_params = dict( winSize  = (5,5), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Calculate optical flow between first and second frame
# The function implements a sparse iterative version
# of the Lucas-Kanade optical flow in pyramids.
# We pass: first image, next image, first points, parameters
# We get:
# - next points
# - status/match: 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
# - error values
pts_2_of, match, err = cv2.calcOpticalFlowPyrLK(gray_1, gray_2, pts_1, None, **lk_params)

# Select good matching points between the two image frames
good_new = pts_2_of[match==1]
good_old = pts_1[match==1]

# Create a mask image for drawing (u,v) vectors on top of the second frame
mask = np.zeros_like(frame_2)

# Draw the lines between the matching points (these lines indicate motion vectors)
for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    # draw points on the mask image
    mask = cv2.circle(img=mask, center=(a,b), radius=5, color=(200), thickness=-1)
    # draw motion vector as lines on the mask image
    mask = cv2.line(img=mask, pt1=(a,b), pt2=(c,d), color=(200), thickness=3)
    # add the line image and second frame together

# Overlay mask
composite_im = np.copy(frame_2)
composite_im[mask!=0] = [0]

# It doesn't seem to work that well; maybe the movement was too big
plt.imshow(composite_im)
```

## 2. Robot Localization

## 3. Mini-Project: 2D Histogram Filter

## 4. Introduction to Kalman Filters

## 5. Representing State and Motion

## 6. Matrices and Transformation of State

## 7. Simultaneous Localization and Mapping

## 8. Vehicle Motion Calculus (Optional)

## 9. Project: Landmark Detection & Tracking (SLAM)

See project repository: [slam_2d](https://github.com/mxagar/slam_2d).

