# AWS SageMaker Studio Lab

This document collects tips and tricks related to the usage of AWS SageMaker Studio Lab, which is a simplified version of AWS SageMaker Studio, but still with GPU access in the free-tier.

Table of contents:

- [AWS SageMaker Studio Lab](#aws-sagemaker-studio-lab)
  - [Introduction](#introduction)
  - [Usage](#usage)

## Introduction

AWS SageMaker Studio Lab is equivalent to Google Colab, but with the following differences:

- It is free.
- The UI is Jupyter lab itself, so we can install plugins, etc.
- We have access to the Terminal.
- We can create our own environments.
- The data persists even after closing the session; storage of 15 GB.
- We need to request access to it.
- We can also connect Github repositories.
- We can have one active project at a time.

AWS SageMaker Studio Lab accounts are separate from AWS accounts, and we need to request one:

[https://studiolab.sagemaker.aws/](https://studiolab.sagemaker.aws/)

The access should be granted in 1-5 business days.

If we need more or extended capabilities, we can easily switch to AWS SageMaker Studio.

## Usage

After the registration process we can access AWS SageMaker Studio Lab here:

[https://studiolab.sagemaker.aws](https://studiolab.sagemaker.aws)

There, we can choose our runtime:

- CPU, 8h
- GPU, 4h

Then, a Jupyter notebook is opened, runnig on AWS.

We can:

- Connect/clone a Github repository
- Open a Terminal windows
- Install anything, e.g., a new environment
- Install plugins
- etc.

The default conda environment is `(studiolab)`.

If we want to link to a AWS SageMaker Studio Lab notebook, we can modify and use the following snippet:

```
[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/org/repo/blob/master/path/to/notebook.ipynb)
```

**IMPORTANT: Stop the runtime at the login window when finished!**
