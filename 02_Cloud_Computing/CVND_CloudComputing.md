# Udacity Computer Vision Nanodegree: Introduction

These are my personal notes taken while following the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).

The nanodegree is composed of six modules:

1. Introduction to Computer Vision
2. Cloud Computing (Optional)
3. Advanced Computer Vision and Deep Learning
4. Object Tracking and Localization
5. Extra Topics: C++ Programming

Each module has a folder with its respective notes.
This folder/file refers to the **second** module (optional): **Cloud Computing**.

Note that:

- I made many hand-written nortes, which I will scan and push to this repostory.
- I forked the Udacity repositors for the exercises; all the material and  notebooks are there:
	- [CVND_Exercises](https://github.com/mxagar/CVND_Exercises)
	- [DL_PyTorch](https://github.com/mxagar/DL_PyTorch)

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

1. Cloud Computing with Google Cloud (It did not work!)
2. Cloud Computing with AWS EC2

## 1. Cloud Computing with Google Cloud

**I tried the steps shown in the lesson, but it did not work; probably, because the lesson is out of date. The error I got was related to the permissions for the project.**

I did not spend time trying to make it work.

I directly used thw AWS services instead.

### 1.1 Create a New Project & Define Quotas & Setup Account

Go to Google CLoud Console: [https://console.cloud.google.com/](https://console.cloud.google.com/)

Select the current project on the horizontal menu (`My First Project`).

Then, on the pop up menu: `Create New Project`: `facial-keypoints-1`.

Now, on the hamburger menu: `IAM & Admin` > `Quotas`

Maybe we are requested to set our billing information.

I had to enable the `Compute Engine API`: Hamburger menu, `APIs & Services` > Library: Select `Compute Engine API`. Then, I had to set a billing account with 3-month free trial.

I had to upgrade the account: Hamburger menu, `Bulling` > `Upgrade`.

Again, on the hamburger menu: `IAM & Admin` > `Quotas`: Filter table and select a quota:

- Service: `Compute Engine API`
- Metric: GPU, `compute.googleapis.com/nvidia_k80_gpus`
- Dimensions (location): select one nearby: `europe-west-1`

We can edit quotas, i.e., increase quota limits: `EDIT QUOTAS`

### 1.2 `gcloud` Command Line Interface Tool (CLI) and the **Cloud Shell**

Interactive Installation of the `gcloud` CLI on the local machine:

```bash
# Follow installation
# I installed in on my home (default): ~/google-cloud-sdk
# I chose Y for all questions
curl https://sdk.cloud.google.com | bash
# Restart shell
exec -l $SHELL
# Initialize gcloud: log in as mxagar
gcloud init
# Finihed
```

In addition to the `gcloud` CLI, we also have the **Cloud Shell** on the web. It can be activated by pressing the shell icon on the blue top menu.

Some useful commands of the `gcloud` CLI:

```bash
# List of running instances: avoid costs!
gcloud compute instances list
```

### 1.3 Launch an Instance

First: Request a Quota as described above; basically, we need to have the checkbox of a `Compute Engine API` active.

Second: Open **Cloud Shell** on wbe interface.

Third: Run these commands (with adjusted parameters) on the **Cloud Shell**.

```bash
export INSTANCE_NAME="my-compute-engine-instance" # Put any name here!
export IMAGE="pytorch-latest-cu91-1531880092"
export PROJECT_NAME="facial-keypoints-1" # Put your Google Cloud Project name here!
export ZONE="europe-west1"  # Put zone you requested quota in: asia-east1-a, europe-west1-d or us-west1-b
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image=$IMAGE \
        --project=$PROJECT_NAME \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator='type=nvidia-tesla-k80,count=1' \
        --metadata='install-nvidia-driver=True' \
        --machine-type=n1-standard-2
```

Then you wait! Instance creation will take a few minutes. Keep in mind that instance will NOT be immediately available for accessing via SSH since the NVIDIA Driver is installed during the first boot. Please wait approx. 2 minutes after the instance is created before SSHing to it.

Note: If you try to SSH to the instance before NVIDIA Driver installation is finished, you will see a "Connection refused" error. You just have to wait for this error to resolve itself.

#### Shutdown

You will also have to shutdown this instance when you are not using it otherwise you risk incurring additional charges. To “stop” (i.e. shutdown) your instances, run one of the following commands:

```bash
gcloud compute instances stop $INSTANCE_NAME
```

Or if you want to delete instance completely:

```bash
gcloud compute instances delete $INSTANCE_NAME
```

Deleting is generally recommended, unless you are shortly pausing your work and will return to it in an hour or so (then, stopping is recommended).

### 1.4 Login to the Instance

After waiting for a couple minutes for your instance to launch, you can access it! In order to ssh into your instance run the following command in gcloud:

```bash
gcloud compute ssh --project $PROJECT_NAME --zone $ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080
```

On the instance you now need to install some packages that are required for the course; the following installs a Python wrapper for use of the OpenCV library:

```bash
sudo pip install opencv-python 
sudo pip3 install opencv-python 
```

Finally, you'll need to clone a Github repository. Run the following command to clone the first project repository that has all the project notebooks and resources:

```bash
git clone https://github.com/udacity/P1_Facial_Keypoints
```

Note: These GPU instances only support JupyterLab as opposed to plain Jupyter notebooks. This just means the interface you're used to in the classroom notebooks will be slightly different (and with a nav bar fornavigating between files)!

## 2. Cloud Computing with AWS EC2

### 2.1 Launch EC2 Instances

EC2 = Elastic Compute Cloud. We can launch VM instances.

Create an AWS account, log in to the AWS console & search for "EC2" in the services.

Select region on menu, top-right: Ireland, `eu-west-1`. Selecting a region **very important**, since everything is server region specific. Take into account that won't see the instances you have in different regions than the one you select in the menu! Additionally, we should select the region which is closest to us. Not also that not all regions have the same services and the service prices vary between regions!

Press: **Launch Instance**.

Follow steps:

1. Choose an Amazon Machine Image (AMI) - An AMI is a template that contains the software configuration (operating system, application server, and applications) required to launch your instance. I looked for specific AMIs on the search bar and selected `Deep Learning AMI (Amazon Linux 2) Version 61.3`.
2. Choose an Instance Type - Instance Type offers varying combinations of CPUs, memory (GB), storage (GB), types of network performance, and availability of IPv6 support. AWS offers a variety of Instance Types, broadly categorized in 5 categories. You can choose an Instance Type that fits our use case. The specific type of GPU instance you should launch for this tutorial is called `p2.xlarge`. I asked to inccrease the limit for EC2 in the support/EC2-Limits menu option to select `p2.xlarge`; meanwhile, I chose `t2.micro`, elegible for the free tier.
3. Configure Instance Details - Provide the instance count and configuration details, such as, network, subnet, behavior, monitoring, etc.
4. Add Storage - You can choose to attach either SSD or Standard Magnetic drive to your instance.
5. Add Tags - A tag serves as a label that you can attach to multiple AWS resources, such as volumes, instances or both.
6. Configure Security Group - Attach a set of firewall rules to your instance(s) that controls the incoming traffic to your instance(s). You can select or create a new security group; when you create one:
	- Select: Allow SSH traffic from anywhere
	- Then, when you launch the instance, you edit the security group
7. Review - Review your instance launch details before the launch.
8. I was asked to create a key-pair; I created one with the name `face-keypoint-detection`using RSA. You can use a key pair to securely connect to your instance. Ensure that you have access to the selected key pair before you launch the instance. A file `face-keypoint-detection.pem`was automatically downloaded.

More on [P2 instances](https://aws.amazon.com/ec2/instance-types/p2/)

Edittting the security group: left menu, `Network & Security` > `Security Groups`:

- Select the security group associated with the created instance (look in EC2 dashboard table)
- Inbound rules (manage/create/add rule): SSH, 0.0.0.0/0, Port 22
- Outbound rules (manage/create/add rule):
	- SSH, 0.0.0.0/0, Port 22
	- Jupyter, 0.0.0.0/0, Port 8888

**Important: Always shut down / stop all instances if not in use to avoid costs! We can re-start afterwards!**. AWS charges primarily for running instances, so most of the charges will cease once you stop the instance. However, there are smaller storage charges that continue to accrue until you **terminate** (i.e. delete) the instance.

We can also set billing alarms.

### 2.2 Connect to an Instance



