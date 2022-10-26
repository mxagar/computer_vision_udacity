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
- Obstacle avoidance
- Eye tracking
- etc.

### 1.2 Motion Vectors
![Motion Vectors](./pics/motion_vectors.jpg)

## 2. Robot Localization

## 3. Mini-Project: 2D Histogram Filter

## 4. Introduction to Kalman Filters

## 5. Representing State and Motion

## 6. Matrices and Transformation of State

## 7. Simultaneous Localization and Mapping

## 8. Vehicle Motion Calculus (Optional)

## 9. Project: Landmark Detection & Tracking (SLAM)

See project repository: [slam_2d](https://github.com/mxagar/slam_2d).

