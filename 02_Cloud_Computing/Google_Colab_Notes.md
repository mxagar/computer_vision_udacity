# Google Colab Notes

This document collects tips and tricks related to the usage of Google Colab.

Overview of Contents:

- [Google Colab Notes](#google-colab-notes)
  - [Google Colab Pro Features](#google-colab-pro-features)
  - [Basic Usage](#basic-usage)
    - [Colab 101](#colab-101)
    - [Check GPU we have](#check-gpu-we-have)
    - [How much memory are we using?](#how-much-memory-are-we-using)
    - [Upload files from desktop to Colab environment](#upload-files-from-desktop-to-colab-environment)
    - [Download files from Colab environment to desktop](#download-files-from-colab-environment-to-desktop)
    - [Mount Google Drive in Colab environment](#mount-google-drive-in-colab-environment)
    - [Installing libraries that are not in Colab](#installing-libraries-that-are-not-in-colab)
    - [Download files to Colab environment](#download-files-to-colab-environment)
    - [Github integration](#github-integration)
    - [Example: Persist during training a model checkpoint and re-use it in a later session](#example-persist-during-training-a-model-checkpoint-and-re-use-it-in-a-later-session)
    - [Links](#links)


## Google Colab Pro Features

- Access to more powerful GPUs: speed, RAM, etc.; e.g.: V100 or A100 Nvidia GPU
- 12€/month
- 24h running without interruption
- Terminal usage


## Basic Usage

### Colab 101

Quick start:

- We can create a Colab notebook from Google Drive, too
- When we create a new notebook, it's not connected to a VM, we need to first run something
- Notebooks appear in Google Drive: `Colab Notebooks` 
- We can share notebooks: Up-right, `Share`
- Storage in the Colab environment is temporary
  - If we restart the runtime, the files persist
  - If we change the runtime, the files are removed!

Change GPU/CPU, select standard/high-end memory, etc

    Runtime > Change runtime type

Command Palette

    Shift + Cmd + P

Terminal: very important!

    Shell icon on the left vertical menu bar.
    BUT: now, available for paying users only.

Use Jupyter system aliases!

    !ls
    !pip
    !apt-get
    !pwd
    !mkdir
    !cp
    ...

See/Stop what's running

    Runtime > Manage Sessions

Billing

> If you encounter errors or other issues with billing (payments) for Colab Pro, Pro+ or pay as you go, please email colab-billing@google.com

### Check GPU we have

```python
# Basic info
!nvidia-smi

# Processed info text
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
```

### How much memory are we using?

```python
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')
```

### Upload files from desktop to Colab environment

```python
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
```

### Download files from Colab environment to desktop

```python
from google.colab import files

# Create an example file (not really necessary)
with open('example.txt', 'w') as f:
  f.write('some content')

# Download a file, eg a *.pkl or a *.pt
files.download('example.txt')
```

### Mount Google Drive in Colab environment

```python
# Mount to Colab folder 
from google.colab import drive
drive.mount('/content/drive')

# Write a file to Google Drive
with open('/content/drive/MyDrive/Projects/foo.txt', 'w') as f:
  f.write('Hello Google Drive!')
!cat /content/drive/MyDrive/Projects/foo.txt

# We can copy things to a local folder!
# Open the Terminal
cp -r drive/MyDrive/Data/FaceKeypoints .

# Flush and unmount; changes persist in Google Drive
drive.flush_and_unmount()
print('All changes made in this colab session should now be visible in Drive.')
```

### Installing libraries that are not in Colab

Use either `pip` or `apt-get`

    !pip install matplotlib-venn
    !apt-get -qq install -y libfluidsynth1

```python
# Install 7zip reader libarchive
# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive

# Install GraphViz & PyDot
# https://pypi.python.org/pypi/pydot
!apt-get -qq install -y graphviz && pip install pydot
import pydot

# Install cartopy
# https://scitools.org.uk/cartopy/docs/latest/
!pip install cartopy
import cartopy
```

### Download files to Colab environment

We can use `wget` and `unzip`.

See [How To Use Wget to Download Files and Interact with REST APIs](https://www.digitalocean.com/community/tutorials/how-to-use-wget-to-download-files-and-interact-with-rest-apis).

```python
# wget options
# -O output filename
!wget -O jquery.min.js https://code.jquery.com/jquery-3.6.0.min.js
# -P output directory
!mkdir Downloads
!wget -P Downloads/  https://code.jquery.com/jquery-3.6.0.min.js
# Show progress
!wget -q --show-progress https://code.jquery.com/jquery-3.6.0.min.js 

# Unzip
# -q: quiet, less output
!unzip -q filename.zip
# Unzip to directory
!unzip filename.zip -d /path/to/directory
```

### Github integration

[Using Google Colab with GitHub](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

We can open Github notebooks from Colab, not only ours, but any public!

    File > Open
    Choose Github
    Choose username: can be any!
    Choose repo + branch: can be any public!
    Open notebook :)

Also, it is possible to directly use the browser:

    Base address 
        https://colab.research.google.com/github/
    Github notebook example
        https://github.com/mxagar/dermatologist-ai/blob/master/dataset_structure_visualization.ipynb
    Colab address
        https://colab.research.google.com/github/<USER>/<PATH-TO-NOTEBOOK>
        https://colab.research.google.com/github/mxagar/dermatologist-ai/blob/master/dataset_structure_visualization.ipynb

We can also open private repositories if we have access:

    File > Open
    Choose Github
    Choose username: our username
    Check: Include private > Click to grant access
    Choose repo + branch: can be any private now!
    Open notebook :)

We can save any Colab notebook on Google Drive or Github; if we want to save changes of a Github notebook opened in Colab we need to grant push permissions to Colab

    File > Save a Copy in Drive
    File > Save a Copy to Github

We can use the Open in Colab shield!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mxagar/dermatologist-ai/blob/master/dataset_structure_visualization.ipynb)

### Example: Persist during training a model checkpoint and re-use it in a later session

TBD.

### Links

- [Overview of Colaboratory Features (Beginner)](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
- [Saving Notebooks To GitHub or Drive](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
- [Snippets: Importing libraries](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb)
- [External data: Local Files, Drive, Sheets and Cloud Storage, IO](https://colab.research.google.com/notebooks/io.ipynb)

