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
    - [2.1 Review of Probability](#21-review-of-probability)
      - [Probability Distributions and Bayes' Rule](#probability-distributions-and-bayes-rule)
    - [2.2 Probabilistic Localization](#22-probabilistic-localization)
    - [2.3 Robot Localization Notebooks](#23-robot-localization-notebooks)
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

### 2.1 Review of Probability

![Review of Probability](./pics/Probability.jpg)

#### Probability Distributions and Bayes' Rule

The Bayes' rule:

`P(A|B) = P(A) * P(B|A)/P(B)`

- `P(A|B)`: probability of occurring event A given B is true; **posterior**.
- `P(A)`, `P(B)`: probabilities of observing A or B without any conditions; `P(A)` is the **prior**.
- `P(B|A)`: probability of occurring B given A is true.

In many situations the event is fixed, so we omit `P(B)` and the rule becomes:

`P(A|B) proportional to P(A) * P(B|A)`

In other words, the posterior is proportional to the prior times the likelihood.

We can model location vectors as probability distributions; the shape of the distribution tells us the most likely location values; i.e., peaks of higher density values denote most likely values.

At the beginning, our distributions might be quite flat, but as we get more data/inputs, we can produce more meaningful distributions, i.e., distributions with more salient peaks. The Bayes' rule is the mathematical tool to achieve that.

Practical interpretation of the Bayes' Rule: given an initial prediction, if we gather additional data (related to our prediction), we can improve the prediction.

Example: we have GPS data of the car location; however, that location has an uncertainty region of 5m. If we collect data from more sensors, we can produce a better estimate of the car location by combining all sensor inputs.

In the Bayesian framework, we have:

- Prior: a prior probability distribution of an uncertain quantity (e.g., location); this is our belief before any measurement.
- Posterior: updated probability distribution of the same quantity after some evidence, i.e., measurements.

### 2.2 Probabilistic Localization

Let's consider a robot which is trying to find out where it is.

- First, it doesn't know: belief probability distribution is flat.
- It measures a door; since it knows the map, the location probability distribution changes: we have 3 bumps where doors are; that's our posterior: an updated belief of where the robot could be.
- Then, the robot moves to the right: the probability distributions are shifted to the right, too, using a convolution. We shift the probability distribution because that is the map we think where the robot is; if the robot moves in a direction, we need to move our current location map accordingly. A convolution is applied because we apply the uncertainty of the movement, i.e., we flatten the distribution according to the probability of moving correctly/wrong.
- The robot senses a door; where is it? If we multiply the posterior with its convoluted we get the probability distribution of the location.

![Probabilistic Localization](./pics/probabilistic_localization.jpg)

### 2.3 Robot Localization Notebooks

Exercise repository: [CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises).

There are several notebooks in the folder: `4_2_Robot_Localization`.

The final that summarizes it all is `9_1. Sense and Move, exercise.ipynb`.

Basically, a **histogram filter** (aka. Monte Carlo Localization) is built step by step in all the notebooks: 

> Histogram filters decompose the state space into finitely many regions and represent the cumulative posterior for each region by a single probability value. When applied to finite spaces, they are called discrete Bayes filters; when applied to continuous spaces, they are known as histogram filters. [Probabilistic Robotics](https://calvinfeng.gitbook.io/probabilistic-robotics/basics/nonparametric-filters/01-histogram-filter)

We have the following scenario:

- A world composed by 5 cells which can be `red` / `green`.
- We **have a map of the world**.
- A robot which can **move** in the world and can **sense** the color of the cell it is in.
- The world is cyclical: if we keep forward in the last cell we appear in the first.

The **goal is: localize the robot in the world map as it moves and senses.**

![Robot Sensing: Scenario](./pics/robot_sensing.png)

The localization problem is solved with a sense-move cycle:

1. First, the probability of the robot being in any of the 5 cells is `1/5 = 0.2`: `p = [0.2, 0.2, 0.2, 0.2, 0.2]`. That is our first uninformative **initial belief or prior**, the one with the highest entropy (i.e., less possible information).
2. Then, we **sense the cell color we're in**, e,g, `red`. Any measurement has a probability of being right (`pHit`) / wrong (`pMiss`). Thus, we update our localization probability map by multiplying to the previous probability values 
   - `pHit` if the world-cell has the same color as the reading (`red`)
   - `pMiss` if the world-cell has a different color than the reading (`green`)
3. Then, we **move the robot a step size in a direction**, e.g., `1 right`. Again, the movement is not perfect, we have an inaccuracy; as such, we define the probabilities of being exact (`pExact`), of undershooting (`pUndershoot`) and of overshooting (`pOvershoot`). The movement updates the localization probability map with a [convolution](https://en.wikipedia.org/wiki/Convolution): the distribution is shifted in the direction of the motion while applying the uncertainty of the movement. That yields a new updated posterior distribution!
4. Then, we repeat 2-3 again indefinitely. After each cycle, i.e., after each `move()`, we have an updated **posterior**.

![Robot Sensing: Scenario](./pics/sense-move.png)

![Robot Sensing: Probability Localization](./pics/probabilistic_localization.jpg)

Note that:

- Every time we **measure** the **entropy decreases**; i.e., we have **gained information** of where the robot is. Thus, the localization distribution has more clear peaks.
- Every time we **move** the **entropy increases**; i.e., we have **lost information** of where the robot is. Thus, the localization distribution is flatter.
- When we move in a direction, we basically shift our belief map in that direction with the robot; however, since the movement has an uncertainty, we need to account for it. That's why we use a convolution in which the previous belief is convoluted with the uncertainty in the direction of the movement.

The **entropy** is measured as: `- sum(p*ln(p))`.

Case of maximum entropy: `p = [0.2, 0.2, 0.2, 0.2, 0.2] -> E = -5*0.2*ln(
0.2) = 0.699`.

All that is summarized in the following lines of code:

```python
import matplotlib.pyplot as plt
import numpy as np

def display_map(grid, bar_width=1):
    if(len(grid) > 0):
        x_labels = range(len(grid))
        plt.bar(x_labels, height=grid, width=bar_width, color='b')
        plt.xlabel('Grid Cell')
        plt.ylabel('Probability')
        plt.ylim(0, 1) # range of 0-1 for probability values 
        plt.title('Probability of the robot being at each cell in the grid')
        plt.xticks(np.arange(min(x_labels), max(x_labels)+1, 1))
        plt.show()
    else:
        print('Grid is empty')

# INITIAL BELIEF: completely uncertain
p = [0.2, 0.2, 0.2, 0.2, 0.2]
# WORLD MAP: the color of each grid cell in the 1D world
world=['green', 'red', 'red', 'green', 'green']
# Measurement = Z, the sensor reading ('red' or 'green')
measurements = ['red', 'red'] # sequence of mesurements
pHit = 0.6 # the probability that it is sensing the color correctly
pMiss = 0.2 # the probability that it is sensing the wrong color

motions = [1,1] # sequence of movement steps to the right
# Movement accuracy
pExact = 0.8 # probability of moving correctly
pOvershoot = 0.1
pUndershoot = 0.1

def sense(p, Z):
    ''' Takes in a current probability distribution, p, and a sensor reading, Z.
        Returns a *normalized* distribution after the sensor measurement has been made, q.
        This should be accurate whether Z is 'red' or 'green'. '''
    q=[]
    # Loop through all grid cells
    for i in range(len(p)):
        # Check if the sensor reading is equal to the color of the grid cell
        # if so, hit = 1
        # if not, hit = 0
        # Basically, we apply pHit in all map cells
        # with the measurement value, pMiss in the rest
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
        
    # Normalize: divide all elements of q by the sum
    # because the complete distribution should add up to 1
    s = sum(q)
    for i in range(len(p)):
        q[i] = q[i] / s
    return q


def move(p, U):
    q=[]
    # Iterate through all values in p
    # The localization probability map is shifted
    # in the direction of motion and the accuracy
    # of the movement is also applied.
    # This is a CONVOLUTION of the previous p map
    for i in range(len(p)):
        # use the modulo operator to find the new location for a p value
        # this finds an index that is shifted by the correct amount
        index = (i-U) % len(p)
        #nextIndex = (index+1) % len(p)
        #prevIndex = (index-1) % len(p)
        nextIndex = (i-U+1) % len(p)
        prevIndex = (i-U-1) % len(p)
        s = pExact * p[index]
        s = s + pOvershoot  * p[nextIndex]
        s = s + pUndershoot * p[prevIndex]
        # append the correct, modified value of p to q
        q.append(s)
    return q

# Compute the posterior distribution if the robot first senses red, then moves 
# right one, then senses green, then moves right again, starting with a uniform prior distribution.
# This loop is in reality valid for any sequence of measurements and motions
for i in range(len(measurements)):
    p = sense(p, measurements[i])
    p = move(p, motions[i])
# Print/display that distribution
print(p)
display_map(p, bar_width=0.9)
```

## 3. Mini-Project: 2D Histogram Filter

This section is about translating the previous toy example in from the 1D world to the 2D world.

There are no videos / instructions, only the following self-assessed project:

[CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises) ` / 4_3_2D_Histogram_Filter`

The project contains the following files:

- `writeup.ipynb`: the project notebook in which all the other files are used.
- `helpers.py`: auxiliary functions, such as `normalize()` and `blur()`
- `localizer.py`: the histogram filter is implemented here: `sense()` and `move()` functions are implemented for the 2D world.
- `simulate.py`: the class `Simulation` is defined, which instantiates a 2D world and enables a simulated movement in it.

The project is about

- implementing `sense()` from `localizer.py`
- and fixing a bug in `move()` from `localizer.py`.

I had to modify some other lines to update the code to work with Python 3.

In the following, the code from `localizer.py`:

```python
#import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

def sense(color, grid, beliefs, p_hit, p_miss):
    new_beliefs = []

    # Loop through all grid cells
    for i, row in enumerate(beliefs):
        p_row_new = []
        for j, cell in enumerate(row):
            # Check if the sensor reading is equal to the color of the grid cell
            # if so, hit = 1
            # if not, hit = 0
            # Basically, we apply pHit in all map cells
            # with the measurement value, pMiss in the rest
            hit = (color == grid[i][j])
            # Save column/cell in row
            p_row_new.append(beliefs[i][j] * (hit * p_hit + (1-hit) * p_miss))
        # Save row in grid
        new_beliefs.append(p_row_new)
        
    # Normalize: divide all elements of new_beliefs by the sum
    # because the complete distribution should add up to 1
    new_beliefs = normalize(new_beliefs)
            
    return new_beliefs

def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    new_G = [[0.0 for i in range(width)] for j in range(height)]
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            # FIXED: width <-> height were interchanged,
            # which had an effect only with rectangular grids 
            new_i = (i + dy ) % height # width
            new_j = (j + dx ) % width # height
            #print("width, height:", width, height)
            #print("i, j:", i, j)
            #print("dy, dx:", dy, dx)
            #print("new_i, new_j:", new_i, new_j)
            #pdb.set_trace()
            new_G[int(new_i)][int(new_j)] = cell
    return blur(new_G, blurring)

```

## 4. Introduction to Kalman Filters

> :warning: I think this section is quite introductory; I suggest reading my hand written notes on the Kalman filter: [KalmanFilter_Notes_2020.pdf](KalmanFilter_Notes_2020.pdf).

The **Kalman Filter** is similar to the **histogram filter** (aka. Monte Carlo Localization), and it is used also to track or localize objects.

The main differences are:

- Kalman works with continuous variables and uni-modal representations (= they have one peak).
- The histogram filter works with discrete variables and multi-modal representations (= they can have several peaks).

![Kalman and Histogram](./pics/kalman_histogram.jpg)

In a discrete world, we use histograms: the variable space is divided in bins and we give a probability to each bin; the resulting histogram is an approximation of the underlying distribution.

In a continuous world we use Gaussians to represent variables; the area below the Gaussian is 1.

![Kalman and Histogram](./pics/kalman_and_gaussian.jpg)

### 4.1 Gaussian Representations and the Update Step

Any Gaussian can be parametrized by 

- its center `mu`, i.e., the mean, where its maximum or peak is,
- and its spread `sigma^2`, i.e., the variance.

Additionally, Gaussians

- are symmetrical
- have only one peak = they're uni-modal

Of course, we'd like to have the smallest `sigma` possible, because that is associated with a variable with less uncertainty.

![Gaussian](./pics/gaussian.jpg)

Similarly as we were talking about the cycle in histogram filters composed by two steps (move and sense), we also have a cycle with two steps in Kalman filters:

1. Measurement and Update (implemented as multiplications between Gaussians)
2. Prediction (implemented as a convolution or an addition of Gaussians)

In both cases, we work with variables represented as Gaussians.

![Gaussian](./pics/kalman_filter_cycle.jpg)

In the case of the measure and update step we have two Gaussians:

- a prior belief
- and a measurement.

In order to get an updated distribution / Gaussian, we simply multiply both Gaussians. In that product, 

- the new mean is shifted to the distribution with less spread
- and the new spread is smaller than the previous two!

![Gaussian Update: Product](./pics/gaussian_update.png)

We need to multiply the two Gaussians because we're applying the Bayes' rule: we're obtaining a posterior from a prior after we measure a conditional event.

As shown in the next figure:

- The new mean is the sum of the two means weighted by the opposite variance.
- The new variance is the inverse of the sum of the inverse variances; that's why it is smaller than any of them (it's like two springs in parallel):

![Gaussian Update: Product](./pics/bayes_multiplication.jpg)

### 4.2 Gaussian Update Step: Notebooks

The following notebooks are very simple; basically, Gaussian operations are implemented manually. They refer to the first step in the Kalman filter cycle: *Measure and Update*.

The notebooks can be found in

[CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises) ` / 4_4_Kalman_Filters`

- `1_1. Gaussian Calculations.ipynb`
- `2_1. New Mean and Variance, exercise.ipynb`

Ans this is the code in them:

```python
# import math functions
from math import *
import matplotlib.pyplot as plt
import numpy as np

# gaussian function
def f(mu, sigma2, x):
    ''' f takes in a mean and squared variance, and an input x
       and returns the gaussian value.'''
    coefficient = 1.0 / sqrt(2.0 * pi *sigma2)
    exponential = exp(-0.5 * (x-mu) ** 2 / sigma2)
    return coefficient * exponential

### ---

# display a gaussian over a range of x values
# define the parameters
mu = 10
sigma2 = 4

# define a range of x values
x_axis = np.arange(0, 20, 0.1)

# create a corresponding list of gaussian values
g = []
for x in x_axis:
    g.append(f(mu, sigma2, x))

# plot the result 
plt.plot(x_axis, g)

### ---

# the update function
def update(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.'''
    new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
    new_var = 1/(1/var2 + 1/var1)
    
    return [new_mean, new_var]
```

### 4.3 Prediction Step

The prediction step consists in observing the motion and forecasting the next state based on it. It is equivalent to what we did with the histogram filter: We need to take the Gaussian and shift it in the direction of the movement; additionally, the uncertainty of the movement needs to be taken into account.

The operation that accomplishes all that is the convolution; however, **in the case of Gaussians, that consists in adding to the parameters of the distribution the movement value and the variance of the movement**:

- `mu_new <- mu_old + u`
- `sigma^2_new <- sigma^2_old + r^2`

![Kalman Filter: Motion Update](./pics/motion_update.jpg)

```python
# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    ## TODO: Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2
    
    return [new_mean, new_var]
```

### 4.4 1D Kalman Filter

This section integrates the `update()` and `predict()` functions for Gaussian states (in 1D) to the Kalman filter cycle: *Measure and Update*.

The notebook can be found in

[CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises) ` / 4_4_Kalman_Filters / 4_1. 1D Kalman Filter, exercise.ipynb`


```python

# the update function
def update(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.'''
    # Calculate the new parameters
    new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
    new_var = 1/(1/var2 + 1/var1)
    
    return [new_mean, new_var]


# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    # Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2
    
    return [new_mean, new_var]

### ---

# measurements for mu and motions, U
measurements = [5., 6., 7., 9., 10.]
motions = [1., 1., 2., 1., 1.]

# initial parameters
measurement_sig = 4. # measurement uncertainty, constant
motion_sig = 2. # motion uncertainty, constant
mu = 0. # initial location estimation
sig = 10000. # initial location uncertainty (high confusion)

## Loop through all measurements/motions
## and print out and display the resulting Gaussian 
## Note that even though the initial estimate for location
## (the initial mu) is far from the first measurement,
## it catches up quickly as we cycle through measurements and motions.
for i in range(len(measurements)):
    # measure and update
    mu, sig = update(mu, sig, measurements[i], measurement_sig)
    print(f'Update: {mu}, {sig}')
    # move and predict
    mu, sig = predict(mu, sig, motions[i], motion_sig)
    print(f'Predict: {mu}, {sig}')
    # plot
    x_axis = np.arange(-20, 20, 0.1)
    g = []
    for x in x_axis:
        g.append(f(mu, sig, x))
    plt.plot(x_axis, g)
plt.show()

# Update: 4.998000799680128, 3.9984006397441023
# Predict: 5.998000799680128, 5.998400639744102
# Update: 5.999200191953932, 2.399744061425258
# Predict: 6.999200191953932, 4.399744061425258
# Update: 6.999619127420922, 2.0951800575117594
# Predict: 8.999619127420921, 4.09518005751176
# Update: 8.999811802788143, 2.0235152416216957
# Predict: 9.999811802788143, 4.023515241621696
# Update: 9.999906177177365, 2.0058615808441944
# Predict: 10.999906177177365, 4.005861580844194
```

### 4.5 Going Beyond 1D

Usually we work on n-D worlds, at least 2D, also known as the *state space*. Additionally, **Kalman filters measure position/location, but they are able to implicitly infer the velocity or change rate of the state!**

In order to make that possible, we need to define **motion models**.

## 5. Representing State and Motion

Kalman filters are widely used in robotics for localization because they are able to produce accurate estimates of the state (position and velocity). That happens in a cycle of two steps:

- Measurement update
- Motion prediction (also known as time update)

Moreover, **the state estimate has less uncertainty than any of the two steps: the measurement or the motion**, in other words, Kalman filters effectively filter out uncertainty in the state estimate. That happens because the product of two Gaussians (as happens in the measurement step) has a smaller variance than the original distributions that are multiplied.

![Kalman Filter: Concept](./pics/kalman_filter_concept.jpg)

> The Takeaway: The beauty of Kalman filters is that they combine somewhat inaccurate sensor measurements with somewhat inaccurate predictions of motion to get a filtered location estimate **that is better than any estimates that come from only sensor readings or only knowledge about movement.**

### 5.1 Introduction to State and Motion Models

We can define inside the state vector anything we consider important; usually the **position and velocity** are taken to define the state of a system.

The idea of taking the velocity is based on the fact that we can use it to **predict the next state** based on a **motion model**:

```python
# Motion model: constant velocity
def predict_state(state, dt)
  # The position component is updated linearly with the velocity
  # new_pos = old_pos + old_vel*time
  state = [state[0]+state[1]*dt, state[1]]
  return state

# Initial state
x_0 = 4
vel_0 = 1
state = [x_0, vel_0]
# Time step
dt = 1
# Simulate the next 10 time steps
for i in range(10):
  state = predict_state(state, dt)
```

We can have different motion models, not only the one with the assumed constant velocity; all models have assumptions and all models have errors, e.g., due to wind effects, slope, tire slippage, etc.

Another motion model could consider non-constant velocity, i.e., we include the **acceleration**:

`change in velocity = dv = a * t = acceleration * time`
 
If we apply the equations and integrate the position `p` for a time interval of `dt`:

- Velocity: `v = v_0 + a*dt`
- Position: `p = p_0 + v_0*dt + 0.5*a*dt^2`

In that case, the state must contain the acceleration; the state has the smallest representation possible so that the model works.

The videos explain these kinematics/motion formulae like for kids with images.

![Kinematics: Acceleration and Velocity](./pics/kinematics_acceleration.png)

### 5.2 Car and Color Objects

This section introduces a `Car` class, defined in 

[CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises) ` / 4_5_State_and_Motion car.py`

```python
class Car:
  # p and v are 2D and are stored in attribute state; world is a 2D empty grid
  def __init__(self, position, velocity, world):
    pass
  def move(self, dt=1): # move in the direction of the velocity; record a path; velocity remains constant
    pass
  def turn_left(self): # rotate velocity values
    pass
  def display_world(self): # display world grid, location (X) and path (., older locations)
    pass
```

The interaction with the class is shown in `1. Interacting with a Car Object.ipynb` and successive notebooks.

For instance, moving a car in a 4x4 square:

```python
# Define world grid: 
# y rows (ascending as we go down),
# x cols (ascending as we go right)
height = 4
width = 6
world = np.zeros((height, width))
# Initial State: position + velocity
initial_position = [0, 0] # [y, x] (top-left corner)
velocity = [1, 0] # [vy, vx] (moving downwards)

# Create car in initial position (0,0)
carla = car.Car(initial_position, velocity, world)

# Movement
carla.move()
carla.move()
carla.move()
carla.turn_left()
carla.move()
carla.move()
carla.move()
carla.turn_left()
carla.move()
carla.move()
carla.move()
carla.turn_left()
carla.move()
carla.move()
carla.move()

# Display
carla.display_world()
# Get state
carla.state # [[0, 0], [0, -1]]
```

#### Modifications

Very simple modifications are done in the folder files:

- Implement a function `turn_right()`.
- Add a car color to `__init__()`.
- Implement a `Color` class.
- Operator overloading: `__add__()`.

They assume the user doesn't know object oriented programming.

### 5.3 State: Matrix Notation

The **state transformation matrix** is shown:

![State Transformation Matrix in 1D](./pics/state_transformation_matrix.png)

In my hand written notes it's `F_k`.

![State Transformation Matrix in 2D](./pics/state_transformation_matrix_2d.png)


## 6. Matrices and Transformation of State: Kalman Filter in 2D

If we extend our 1D world to higher dimensions, we need to work with **multivariate Gaussians**, which have

- a vector of means
- and a covariance matrix.

![Multivariate Gaussian](./pics/multivariate_gaussian.jpg)

The covariance matrix has the variance (i.e., uncertainty) of each dimension in its diagonal; if the Gaussian is tilted, there is a **correlation** between the variables, and the covariance matrix is not diagonal, i.e., the values **out** of its diagonal account for those correlations and they are called covariances. That correlation is very informative, since we can estimate where one variable will be if we know the value of another with which it is correlated.

![2D Gaussian](./pics/2d_gaussian.jpg)

In a Kalman filter, our multi-dimensional variable is the **state**; that state is usually composed by the **location** and the **velocity**. Since we have a **motion model**, we know that both are related, and that relationship can be represented as a tilted Gaussian, `F`. Every time we measure the location/position, we also get a Gaussian which has no information about the velocity, but **multiplying both Gaussians, the observed location and the motion model**, we get a very good estimate of the state, which contains all variables!

![Kalman: Prediction](./pics/kalman_prediction.jpg)

:warning: The formula in the figure is wrong; the motion model should be `x' <- x + dt*v`.

Note that we don't measure the velocity, but the estimated state has a value of it with high certainty. In general, we can say that the Kalman filter has two types of variables:

- Observable (location)
- Hidden (velocity)

The Kalman filter is especially good at inferring the value of the hidden variables.

![Kalman Filter: Observable and Hidden Variables](./pics/kalman_variables.jpg)

All in all, the computations to obtain a new state estimate after the measurements `x'` and its associate uncertainty covariance matrix `P'` are the following:

![Kalman Filter: Equations](./pics/kalman_equations.jpg)

That's the Kalman Filter!

We don't need to learn the equations, but these are basically a generalization of the 1D case for multi-dimensional Gaussians. My hand written notes explain them better than here.

Other notations are usually common, too:

![Kalman Filter: Equations](./pics/kalman_equations_notation.jpg)

![Kalman Filter: Equation Notation](./pics/kalman_notation.jpg)

![Kalman Filter: Variable Definitions](./pics/kalman_variable_definitions.jpg)


Notes:

- `x` is the state mean and `P` is the state covariance
- `F` is the state transition matrix, which contains the motion model
- `u` is the control vector, usually the acceleration
- `B` is the control matrix
- `Q` is the uncertainty from the environment added to `P`
- `H` is the measurement/observation matrix which maps the state to the sensor space
- `y` is the error in sensor space between the measurement and the predicted state mapped to the sensor space
- `R` is the uncertainty due to the sensor noise
- `S` is computed to obtain the **Kalman gain** `K`
- The **Kalman gain** yields the new `x'` and its associated covariance matrix `P'`

### Rest of the Section

The rest of the section is in linear algebra, specifically, matrix operations. Very basic content, quite deceiving, but I guess they need to cover all levels. However, it doesn't make much sense to lower the level so much after the *advanced* Deep Learning content.

`x = [x, y, v_x, v_y, phi, alpha]`

Vectors, matrices, etc. are coded; matrix operations are coded by hand.

## 7. Simultaneous Localization and Mapping

Until now, we've seen how to solve localization with two approaches: (1) histogram filters and (2) Kalman filter. In all the cases, the world map was known. SLAM is about simultaneously generating the world map and solving the localization problem.

The idea is that the robot moves and builds the map; when the robot moves, uncertainty due to movement is accumulated, but when the loop is closed (i.e., the robot visits a region it's already been), the uncertainty is reduced and the map is consolidated.

![Mapping: Uncertainty](./pics/mapping_uncertainty.jpg)

![Mapping: Loop closed, uncertainty decreased](./pics/mapping_loop_closed.jpg)

### 7.1 Graph SLAM

Graph SLAM is one method to solve SLAM; it's the easiest to explain.

It consists in collecting:

- The initial pose
- The relative motion poses
- The relative measurement pose of landmarks, done from each new pose

These are the **constrains**; however, they are not rigid, but they are really Gaussians! It's like defining them loosely, like a rubber band. Then, we chain all those constraints in a system and find the most likely concrete values for the system.

![Graph SLAM: Idea](./pics/graph_slam_idea.jpg)

The Gaussian nature of the constraints is explained in the section [7.4 Confidence](#7.4-Confidence).

For more information on the method, check these important articles:

- [Graph SLAM Algorithm, Thrun et al., 2006](http://robots.stanford.edu/papers/thrun.graphslam.pdf)
- [A Tutorial on Graph SLAM Algorithms, Grisetti et al., 20010](http://ais.informatik.uni-freiburg.de/teaching/ws11/robotics2/pdfs/ls-slam-tutorial.pdf)

### 7.2 Constrains

Constraints are implemented as a matrix and a vector in Graph SLAM. The matrix is symmetric and contains all constraint poses in its rows & columns; the vector is analog.

Whenever we move or measure something, the matrix cells/weights are updated additively, i.e., we add the corresponding constraint weights to the previous content.

For each motion/measurement, we update only the region of the matrix which is affected by the motion/measurement.

Example:

    We move from x_0 -> x_1, such that: x_1 = x_0 + 5
    The motion constraint has two variables, so it's re-written in 2 ways: x_0, x_1

      x_0 - x_1 = -5
      x_1 - x_0 = 5
      (-x_0 + x_1 = 5)

    The coefficients of the constraints are transferred to the matrix:

            x_0   x_1 | vector
      x_0   1     -1  |  -5
      x_1   -1    1   |  5

    If we now move again, a new portion of the matrix is updated: x_2 = x_1 - 4

      x_1 - x_2 = 4
      x_2 - x_1 = -4    
      (-x_1 + x_2 = -4)

            x_0   x_1     x_2   | vector
      x_0   1     -1            |  -5
      x_1   -1    1+1=2   -1    |  5+4=9
      x_2         -1      1     |  -4      

    If measure a distance from x_1 to the landmark L_0, say 9

      x_1 - L_0 = -9
      L_0 - x_1 = 9    
      (-x_1 + L_0 = 9)


![Graph SLAM: Constraints](./pics/graph_slam_constrains.jpg)

![Graph SLAM: Constraints](./pics/graph_slam_constrains_2.jpg)

![Graph SLAM: Constraints](./pics/graph_slam_constrains_3.jpg)

Summary:

- For each constraint equation (movement or measurement), that equation is formulated `n` times, being `n` the number of variables `x` or `L`.
- The coefficients of the equations are accumulated (added) in the graph matrix.
- Not every new position must have a landmark measurement!
- The graph matrix might have untouched cells, evaluated with value 0; untouched cells mean that there are no constraints between the variables (`x`, `L`) which those cells relate.
- Since landmarks cannot measure each other, the portion related to the landmarks will be always a diagonal matrix.

That can be summarized as the following algorithm:

    Initial position: x0
      omega[ at x0 ] = 1
      xi[ at x0 ] = initial_pos
    When robot moves dx at xt update the constraint matrices as follows:
      Add [[1, -1], [-1, 1]] to omega at [xt, xt+1]
      Add [-dx, dx] to xi at [xt, xt+1]
      Note: Use Numpy and slicing
    When robot measures dl wrt L at xt update the constraint matrices as follows:
      Add [[1, -1], [-1, 1]] to omega at [xt, L]
      Add [-dl, dl] to xi at [xt, L]
      Note: in contrast to the motion, [xt, L] won't be contiguous usually
      so slicing is not possible; instead, update intersecting rows & cols
    Update related to the confidence:
      The added values should be multiplied by a factor s
      which is really 1/sigma,
      being sigma the spread of the Gaussian.
      In this section s = 1 = 1/stdev, stdev = 1.
      Therefore, we are really summing mean/stdev values!

In the following, small quizes related to which cells are modified/unmodified in given motion systems:

![Graph SLAM: Matrix Modification](./pics/graph_slam_matrix_modification.jpg)

![Graph SLAM: Untouched Cells](./pics/untouched_cells.jpg)

### 7.3 The System: Omega and Xi

The graph representation seen so far is composed by

- The matrix of modified variables, also denoted as `Omega`
- The vectors of modification differences, also called `Xi`

It turns out that the best estimate `mu` for all variables is

`mu = inv(Omega)*Xi`

So we build `Omega` and `Xi` over time and compute `mu` with the formula when we need it.

![Graph SLAM: Omega and Xi](./pics/omega_xi_formula.jpg)

#### Example Notebooks

The notebooks in

[CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises) ` / 4_7_SLAM`

show how different steps of motion and measurement update `Omega` and `Xi`, and how the solution of the system varies depending on the measurements.

The Example taken is the following:

![SLAM Graph Example](./pics/slam_graph_example.png)

Note that the example is 1D! It seems to be 2D because the landmarks and the poses are in a plane, but there is only one coordinate for them, as I understand; in consequence, the solution to the system `x = inv(omega)*xi` yields one coordinate for each variable.

The algorithm followed is the one summarized in the previous section. And the final code to solve it is this:

```python
import numpy as np

# Algorithm:
# 
# Initial position: x0
#   omega[ at x0 ] = 1
#   xi[ at x0 ] = initial_pos
# When robot moves dx at xt update the constraint matrices as follows:
#   Add [[1, -1], [-1, 1]] to omega at [xt, xt+1]
#   Add [-dx, dx] to xi at [xt, xt+1]
#   Note: Use Numpy and slicing
# When robot measures dl wrt L at xt update the constraint matrices as follows:
#   Add [[1, -1], [-1, 1]] to omega at [xt, L]
#   Add [-dl, dl] to xi at [xt, L]
#   Note: in contrast to the motion, [xt, L] won't be contiguous usually
#   so slicing is not possible; instead, update intersecting rows & cols
# Update related to the confidence:
#   The added values should be multiplied by a factor s
#   which is really 1/sigma,
#   being sigma the spread of the Gaussian.
#   In this section s = 1 = 1/stdev, stdev = 1.
#   Therefore, we are really summing mean/stdev values!

def mu_from_positions(initial_pos, move1, move2, Z0, Z1, Z2):
    
    ## Construct constraint matrices
    ## and add each position/motion constraint to them
    
    # Initialize constraint matrices with 0's
    # Now these are 4x4 because of 3 poses and a landmark
    omega = np.zeros((4,4))
    xi = np.zeros((4,1))
    # Variables: x_0, x_1, x_2, L
    # Anothe option:
    # omega = [[0, 0, 0, 0],
    #          [0, 0, 0, 0],
    #          [0, 0, 0, 0],
    #          [0, 0, 0, 0]]
    # xi = [[0],
    #       [0],
    #       [0],
    #       [0]]
    # omega = np.array(omega)
    # xi = np.array(xi)
    
    # Initial pose constraint: x0
    omega[0,0] = 1
    xi[0,0] = initial_pos
    
    # Movement 1
    i = 0 # x0 -> x1
    dx = move1
    omega[i:(i+2),i:(i+2)] += np.array([[1, -1], [-1, 1]])
    xi[i:(i+2),:] += np.array([[-dx], [dx]])
    # Another option:
    # First motion, dx = move1
    # omega += [[1., -1., 0., 0.],
    #           [-1., 1., 0., 0.],
    #           [0., 0., 0., 0.],
    #           [0., 0., 0., 0.]]
    # xi += [[-move1],
    #        [move1],
    #        [0.],
    #        [0.]]

    # Movement 2
    i = 1 # x1 -> x2
    dx = move2
    omega[i:(i+2),i:(i+2)] += np.array([[1, -1], [-1, 1]])
    xi[i:(i+2),:] += np.array([[-dx], [dx]])
    
    ## Sensor measurements for the landmark, L
    
    # Measurement 1
    i = 0 # x0 -> L
    j = 3 # x0 -> L
    dl = Z0
    omega[i,i] += 1
    omega[i,j] += -1
    omega[j,i] += -1
    omega[j,j] += 1
    xi[i,0] += -dl
    xi[j,0] += dl
    # Another option:
    # omega += [[1., 0., 0., -1.],
    #           [0., 0., 0., 0.],
    #           [0., 0., 0., 0.], 
    #           [-1., 0., 0., 1.]]
    # xi += [[-Z0],
    #        [0.0],
    #        [0.0],
    #        [Z0]]

    # Measurement 1
    i = 1 # x1 -> L
    j = 3 # x1 -> L
    dl = Z1
    omega[i,i] += 1
    omega[i,j] += -1
    omega[j,i] += -1
    omega[j,j] += 1
    xi[i,0] += -dl
    xi[j,0] += dl

    # Measurement 3
    i = 2 # x2 -> L
    j = 3 # x2 -> L
    dl = Z2
    omega[i,i] += 1
    omega[i,j] += -1
    omega[j,i] += -1
    omega[j,j] += 1
    xi[i,0] += -dl
    xi[j,0] += dl

    # Display final omega and xi
    print('Omega: \n', omega)
    print('\n')
    print('Xi: \n', xi)
    print('\n')
    
    ## Calculate mu as the inverse of omega * xi
    ## recommended that you use: np.linalg.inv(np.matrix(omega)) to calculate the inverse
    omega_inv = np.linalg.inv(np.matrix(omega))
    mu = omega_inv*xi
    return mu

### ---

# call function and print out `mu`
mu = mu_from_positions(-3, 5, 3, 10, 5, 2)
print('Mu: \n', mu)
# Omega: 
#  [[ 3. -1.  0. -1.]
#  [-1.  3. -1. -1.]
#  [ 0. -1.  2. -1.]
#  [-1. -1. -1.  3.]]
# 
# Xi: 
#  [[-18.]
#  [ -3.]
#  [  1.]
#  [ 17.]]
# 
# Mu: 
#  [[-3.]
#  [ 2.]
#  [ 5.]
#  [ 7.]]

```

### 7.4 Confidence

**What's happening in the background is Gaussian multiplication!** We add coefficients in the `omega` and `xi` matrices, but these are effectively Gaussian multiplications: when Gaussians are multiplied, their exponents are added!

The Gaussian is

    N(m, s) = 1/s*sqrt(2*pi) * exp (0.5*((x - m)/s)^2)

When we go from `x_0` to `x_1` with `dx1`, we have:

    1/s*sqrt(2*pi) * exp (0.5*((x_1 - x_0 - dx1)/s)^2)

When we go from `x_1` to `x_2` with `dx2`, we have:

    1/s*sqrt(2*pi) * exp (0.5*((x_2 - x_1 - dx2)/s)^2)

Making these two steps is like multiplying these two Gaussians, which is equivalent to adding the exponents; if we 

- remove the scaling
- remove the exponent
- add instead of multiply
- take the base of the squared exponent, i.e., remove `^2`

we have *almost* what we've been doing in the `omega` and `xi` matrices; the difference is that we are missing the sigma value, `s`:

    x_1/s - x_0/s = dx1/s
    x_2/s - x_1/s = dx2/s

That means, we have assumed `s = 1` so far; by changing and multiplying the values enetered in `omega` and `xi` we can model the uncertainty; we call `1/s` the strength factor:

- a larger `s` means a smaller strength factor, which is related to less confidence
- a smaller `s` means a larger strength factor, which is related to more confidence.

We basically need to multiply the entered coefficients in `omega` and `xi` and a new solution is obtained, which reveals teh best estimate taking into account the noise/uncertainty:

```python
# Update of the third measurement
# with a very high confidence
strength = 1.0/0.2 # 5
omega += [[0., 0., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., strength, -strength], 
          [0., 0., -strength, strength]]
xi += [[0.],
       [0.],
       [-Z2*strength],
       [Z2*strength]]
```

![Graph SLAM: Confidence](./pics/slam_graph_confidence.jpg)

### 7.5 Graph SLAM: Summary

The following figure summarizes the Graph SLAM approach:

- We add the coefficients of our constraints to the `omega` and `xi` matrices scaled with the `strength = 1 / sigma` factor: initial position, motion, measurement.
- We solve the equation `mu = inv(omega)*xi` and we get the best estimate for all the constraints.

Note that we get both the **path** as well as the **map**!

The module project consists in implementing a 2D slam system.

![Graph SLAM: Summary](./pics/graph_slam_summary.jpg)

### Forum Questions

#### Confidence strength in the xi vector?

Hello,

In the Object Tracking and Localization module of the CVND, concretely in the Chapter "Simultaneous Mapping and Localization", confident measurements are introduced in a video which explains that the omega matrix aggregates really exponential constraints over time; these constraints are in reality scaled by `1/sigma`. Then, Sebastian encourages to increase the confidence of the last measurement (to value 5 = 1/sigma) in an example which is contained in the notebook `3. Confident Measurements.ipynb`.

In that notebook, the following modification is done:

```python
# Strength of 5 applied to coefficients of 1
omega += [[0., 0., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., 5., -5.], 
          [0., 0., -5., 5.]]
xi += [[0.],
       [0.],
       [-Z2],
       [Z2]]
```

However, if I understood correctly, the strength factor should be applied to the `xi` vector, too, because it contains in reality `z/sigma`. Therefore, we should have:

```python
# Strength of 5 applied to coefficients of 1
s = 5.0
omega += [[0., 0., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., s, -s], 
          [0., 0., -s, s]]
xi += [[0.],
       [0.],
       [-Z2*s],
       [Z2*s]]
```
Note that I multiplied the measurement values of `xi` by a factor `s`.

Which version is correct and why: the first or the second?

Thank you,
Mikel

#### Graph SLAM: Dimensions of Constraints

Hello,

There is something a bit confusing in the examples given in the SLAM section of the module "Object Tracking and Localization": the variables of the localization points (x) and landmarks (L) are expressed in 1D world coordinates; however:

- The world coordinate system is not drawn (so we don't know with respect to what we're saying where things are)
- The graph of motion and measurements is shown in a 2D plane even though they should be in a 1D line?

Are my observations correct?

Another related question: if we want to work with a 2D system, we'd have to add rows/columns to `omega` and `xi` to include the new dimensions, right?

For instance:

```python
# 1D
xi = np.array([x0, x1, L1, L2]).T

# 2D
xi = np.array([x0_x, x0_y, x1_x, x1_y, L1_x, L1_y, L2_x, L2_y]).T
```

Is the latter a correct approach?

Thank you,
Mikel


## 8. Vehicle Motion Calculus (Optional)

This section is optional.

It is quite basic, very low level -- I don' understand how/why they do this after all the "advanced" deep learning stuff.

Topics covered:

- Differentiation in code
- Integration in code
- Error Accumulation
- Trajectory computation
- Optional Project: Trajectory Reconstruction from Acceleration Sensor Data


### Odometry

Car navigation sensors:

- Odometer: it measures how far a vehicle has traveled measuring the velocity and the movements of the steering wheel. However, it has an error due to, e.g., wheel diameter. The driver can reset it manually.
- Inertial Measurement Unit (IMU): magnetometer, accelerometer, gyroscope.

### Differenciating from Displacement: Velocity and Acceleration

Velocity = slope of displacement in time.

    v = dx/dt

Acceleration = slope of velocity in time.

    a = dv/dt = d(dx/dt)/dt

The following code differentiates a time series:

```python
# Differentiation function in the helper
def get_derivative_from_data(position_data, time_data):
    """
    Calculates a list of speeds from position_data and 
    time_data.

    Arguments:
      position_data - a list of values corresponding to 
        vehicle position

      time_data     - a list of values (equal in length to
        position_data) which give timestamps for each 
        position measurement

    Returns:
      speeds        - a list of values (which is shorter 
        by ONE than the input lists) of speeds.
    """
    # 1. Check to make sure the input lists have same length
    if len(position_data) != len(time_data):
        raise(ValueError, "Data sets must have same length")

    # 2. Prepare empty list of speeds
    speeds = []

    # 3. Get first values for position and time
    previous_position = position_data[0]
    previous_time     = time_data[0]

    # 4. Begin loop through all data EXCEPT first entry
    for i in range(1, len(position_data)):

        # 5. get position and time data for this timestamp
        position = position_data[i]
        time     = time_data[i]

        # 6. Calculate delta_x and delta_t
        delta_x = position - previous_position
        delta_t = time - previous_time

        # 7. Speed is slope. Calculate it and append to list
        speed = delta_x / delta_t
        speeds.append(speed)

        # 8. Update values for next iteration of the loop.
        previous_position = position
        previous_time     = time

    return speeds

### -- Plot

speeds = get_derivative_from_data(displacements, timestamps)

plt.title("Position and Velocity vs Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Position (blue) and Speed (orange)")
plt.scatter(timestamps, displacements)
plt.scatter(timestamps[1:], speeds)
plt.show()
```

### Integrating Acceleration and Velocity

The integral is the inverse operation of the differentiation. The integral is the area below the curve; as such, we need to define the range in which we compute it.

We can approximate it by computing the area of the small rectangles below the curve

In the following code, several integration code pieces are shown:

```python
# Plot function and the small rectangles,
# i.e., the approximate area below the curve
def show_approximate_integral(f, t_min, t_max, N):
    t = np.linspace(t_min, t_max)
    plt.plot(t, f(t))
    
    delta_t = (t_max - t_min) / N
    
    print("Approximating integral for delta_t =",delta_t, "seconds")
    box_t = np.linspace(t_min, t_max, N, endpoint=False)
    box_f_of_t = f(box_t)
    plt.bar(box_t, box_f_of_t,
            width=delta_t,
            alpha=0.5,
            facecolor="orange",
            align="edge",
            edgecolor="gray")
    plt.show()

# Approximate integral computation
def integral(f, t1, t2, dt=0.1):
    # area begins at 0.0 
    area = 0.0
    
    # t starts at the lower bound of integration
    t = t1
    
    # integration continues until we reach upper bound
    while t < t2:
        
        # calculate the TINY bit of area associated with
        # this particular rectangle and add to total
        dA = f(t) * dt
        area += dA
        t += dt
    return area

# Example function
def f(t):
    return -1.3 * t**3 + 5.3 * t ** 2 + 0.3 * t + 1 

# Plot approximate area below the curve
N = 50
show_approximate_integral(f, 0, 4, N)

# Compute integral
integral(f, 2, 4)

# Integration function in the helper
def get_integral_from_data(acceleration_data, times):
    # 1. We will need to keep track of the total accumulated speed
    accumulated_speed = 0.0
    
    # 2. The next lines should look familiar from the derivative code
    last_time = times[0]
    speeds = []
    
    # 3. Once again, we lose some data because we have to start
    #    at i=1 instead of i=0.
    for i in range(1, len(times)):
        
        # 4. Get the numbers for this index i
        acceleration = acceleration_data[i]
        time = times[i]
        
        # 5. Calculate delta t
        delta_t = time - last_time
        
        # 6. This is an important step! This is where we approximate
        #    the area under the curve using a rectangle w/ width of
        #    delta_t.
        delta_v = acceleration * delta_t
        
        # 7. The actual speed now is whatever the speed was before
        #    plus the new change in speed.
        accumulated_speed += delta_v
        
        # 8. append to speeds and update last_time
        speeds.append(accumulated_speed)
        last_time = time
    return speeds

```

### Rate Gyros

Gyroscopes give us information about the angular velocity; if we integrate it we obtain the angle of the car.

    w = d(r)/dt -> r = int(w)

### Error

We cannot simply integrate / differentiate sensor signals, because they have noise!

If we integrate the position / rotation values, we'll accumulate error over time.

Additionally, any accelerometer has bias: even though the device is not moving/accelerating, they measure small residual acceleration values. Sensor calibration can alleviate this, but it's never completely fixed.

If we perform the calibration, we extract the residual values of the sensor, i.e., the bias. We can use them later in the integration, as follows:

```python
def get_integral_from_data(acceleration_data, times, bias=0.0):
    """
    Numerically integrates data AND artificially introduces 
    bias to that data.
    
    Note that the bias parameter can also be used to offset
    a biased sensor.
    """
    accumulated_speed = 0.0
    last_time = times[0]
    speeds = []
    for i in range(1, len(times)):
        
        # THIS is where the bias is introduced. No matter what the 
        # real acceleration is, this biased accelerometer adds 
        # some bias to the reported value.
        acceleration = acceleration_data[i] + bias
        
        time = times[i]
        delta_t = time - last_time
        delta_v = acceleration * delta_t
        accumulated_speed += delta_v
        speeds.append(accumulated_speed)
        last_time = time
    return speeds
```

### Trajectory Computation

In the notebook

[CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises) ` / 4_8_Vehicle_Motion_and_Calculus / Keeping Track of x and y.ipynb`

the 2D trajectory computation function is implemented in the class `Vehicle`, which is able to *drive* using the functions

- `drive_forward()`
- `turn()`
- `set_heading()`

Then, the function `show_trajectory()` plots the path generated with the used driving commands.

```python
import numpy as np
from math import pi
from matplotlib import pyplot as plt

# these 2 lines just hide some warning messages.
import warnings
warnings.filterwarnings('ignore')

class Vehicle:
    def __init__(self):
        self.x       = 0.0 # meters
        self.y       = 0.0
        self.heading = 0.0 # radians
        self.history = []
        
    def drive_forward(self, displacement):
        """
        Updates x and y coordinates of vehicle based on 
        heading and appends previous (x,y) position to
        history.
        """
        delta_x = displacement * np.cos(self.heading)
        delta_y = displacement * np.sin(self.heading)
        
        new_x = self.x + delta_x
        new_y = self.y + delta_y
        
        self.history.append((self.x, self.y))

        self.x = new_x
        self.y = new_y
    
    def set_heading(self, heading_in_degrees):
        """
        Set's the current heading (in radians) to a new value
        based on heading_in_degrees. Vehicle heading is always
        between -pi and pi.
        """
        assert(-180 <= heading_in_degrees <= 180)
        rads = (heading_in_degrees * pi / 180) % (2*pi)
        self.heading = rads
        
    def turn(self, degrees):
        rads = (degrees * pi / 180)
        new_head = self.heading + rads % (2*pi)
        self.heading = new_head
    
    def show_trajectory(self):
        """
        Creates a scatter plot of vehicle's trajectory.
        """
        # get the x and y coordinates from vehicle's history
        X = [p[0] for p in self.history]
        Y = [p[1] for p in self.history]
        
        # don't forget to add the CURRENT x and y
        X.append(self.x)
        Y.append(self.y)
        
        # create scatter AND plot (to connect the dots)
        plt.scatter(X,Y)
        plt.plot(X,Y)
        
        plt.title("Vehicle (x, y) Trajectory")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()
```

### Optional Project: Trajectory Reconstruction from Acceleration Sensor Data

In this project, we get time-stamped sensor data with the following values:

- Timestamp
- Displacement
- Yaw rate (angular velocity)
- Acceleration (linear)

We need to reconstruct the trajectory of the car.

Many necessary code pieces have been show already, it's an easy project.

The project link:

[CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises) ` / 4_8_Vehicle_Motion_and_Calculus / project_trajectory_reconstruction`

## 9. Project: Landmark Detection & Tracking (SLAM)

See project repository: [slam_2d](https://github.com/mxagar/slam_2d).

