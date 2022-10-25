# Udacity Computer Vision Nanodegree: Personal Notes

These are my personal notes taken while following the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).

The nanodegree is composed of these modules:

1. [Introduction to Computer Vision](01_Intro_Computer_Vision)
2. [Cloud Computing (Optional)](02_Cloud_Computing)
3. [Advanced Computer Vision and Deep Learning](03_Advanced_CV_and_DL)
4. [Object Tracking and Localization](04_Object_Tracking_Localization)

Each module has a folder with its respective notes; **you need to go to each module folder and follow the Markdown file in them**.

Additionally, note that:

- I have also notes on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) in my repository [deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity); that MOOC is strongly related and has complementary material. I have many hand-written notes in [deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity) related to this repository, too.
- The exercises are commented in the Markdown files and linked to their location; most of the exercises are located in other repositories, originally forked from Udacity and extended/completed by me:
	- [CVND_Exercises](https://github.com/mxagar/CVND_Exercises)
	- [DL_PyTorch](https://github.com/mxagar/DL_PyTorch)
	- [CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises)
	- [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch)

## Projects

Udacity requires the submission of a project for each module; these are the repositories of the projects I submitted:

1. Facial Keypoint Detection with Deep Convolutional Neural Networks (CNNs): [P1_Facial_Keypoints](https://github.com/mxagar/P1_Facial_Keypoints).
2. Image Captioning: Image Description Text Generator Combining CNNs and RNNs: [image_captioning](https://github.com/mxagar/image_captioning).
3. Landmark Detection & Tracking (SLAM): [slam_2d](https://github.com/mxagar/slam_2d).

## Practical Installation Notes

You need to follow the installation & setup guide from [CVND_Exercises](https://github.com/mxagar/CVND_Exercises), which can be summarized with the following commands:

```bash
# Create new conda environment to be used for the nanodegree
conda create -n cvnd python=3.6
conda activate cvnd
conda install pytorch torchvision -c pytorch
conda install pip

# Go to the folder where the Udacity DL exercises are cloned/forked,
# after forking the original repo
cd ~/git_repositories/CVND_Exercises
pip install -r requirements.txt
```

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

If you find this repository helpful and use it, please link to the original source.
