# Catalog of Computer Vision Functions Using OpenCV

This file is a short compilation of the most important computer vision (CV) functions.
Most functions are from OpenCV, but some other packages are also considered.
Methods are listed in python; partial in context is also provided in some cases.
Knowledge of CV is assumed - this is basically a list of functions with some sontext examples!

No guaranties.  
Mikel Sagardia, 2022.

## Overview of Contents

- [Catalog of Computer Vision Functions Using OpenCV](#catalog-of-computer-vision-functions-using-opencv)
  - [Overview of Contents](#overview-of-contents)
  - [Imports](#imports)
  - [General](#general)
  - [Numpy: Images stored as matrices!](#numpy-images-stored-as-matrices)
  - [PIL](#pil)
  - [Matbplotlib](#matbplotlib)
  - [OpenCV](#opencv)
  - [Context: Image Representation](#context-image-representation)
  - [Context: Filters, Fourier, Canny, Hough, Haar](#context-filters-fourier-canny-hough-haar)
  - [Context: Harris, Contours, K-Means](#context-harris-contours-k-means)
  - [Context: Pyramids, ORB, HOG](#context-pyramids-orb-hog)
  - [Context: Optical Flow](#context-optical-flow)
  - [Utilities: Re-Usable Snippets](#utilities-re-usable-snippets)
    - [`resize_images()`](#resize_images)

## Imports

```python
```
```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
%matplotlib inline
from PIL import Image
```

## General

```python
type()
print()
eval()
ravel()
k = waitKey(1)
k = ord('q')
```

## Numpy: Images stored as matrices!

```python
np.array()
type()
np.arange()
np.zeros()
np.ones()
np.random.seed() 
arr = np.random.randint()
arr.max()
np.median()
np.max()
np.min()
arr.argmax()
arr.reshape()
mat = np.arange().reshape()
mat[row,col]
mat[:,col] 
mat[start:end,start:end]

np.copy()
np.sum()
np.float32()
np.int0()
im = np.array(...)
im.shape
```

## PIL

```python
pic = Image.open()
im = np.asarray(pic)
plt.imshow(im, cmap='gray')
```

## Matbplotlib

```python
plt.imshow()
plt.subplots()
plt.suptitle()
plt.title()
plt.hist()
plt.plot()
plt.matshow()
ax.annotate()
```

## OpenCV

```python
cv2.imread()
cv2.cvtColor()
cv2.resize()
cv2.flip()
cv2.imwrite()

cv2.inRange()
cv2.equalizeHist()
cv2.calcHist()
cv2.split()
cv2.merge()
cv2.addWeighted()
cv2.bitwise_and()
cv2.bitwise_or()
cv2.bitwise_not()
cv2.filter2D()
cv2.blur()
cv2.GaussianBlur()
cv2.medianBlur()
cv2.bilateralFilter()
cv2.threshold()
cv2.adaptiveThreshold()
cv2.erode()
cv2.dilate()
cv2.morphologyEx()
cv2.Sobel()
cv2.Laplacian()
cv2.Canny()
cv2.findContours()
cv2.drawContours()
cv2.HoughLinesP()
cv2.HoughCircles()
cv2.CascadeClassifier()
cv2.CascadeClassifier.detectMultiScale()
cv2.cornerHarris()
cv2.line()
cv2.rectangle()
cv2.circle()
cv2.polylines()
cv2.fillPoly()
cv2.ellipse()
cv2.getRotationMatrix2D()
cv2.warpAffine()
cv2.add()
cv2.substract()

cv2.accumulateWeighted()
cv2.absdiff()

cv2.moments(cnt)
cv2.contourArea(cnt)
cv2.arcLength(cnt)
cv2.convexHull(cnt)
cv2.isContourConvex(cnt)
cv2.boundingRect(cnt)
cv2.minAreaRect(cnt)
cv2.boxPoints()
cv2.minEnclosingCircle(cnt)
cv2.fitEllipse(cnt)
cv2.fitLine()

cv2.pyrDown()

cap = cv2.VideoCapture(0)
cap.get()
cap.read()
writer = cv2.VideoWriter()
writer.write()
cap.release()
writer.release()
cv2.destroyAllWindows()

cv2.matchTemplate()
cv2.minMaxLoc()

cv2.cornerHarris()
cv2.goodFeaturesToTrack()
cv2.Canny()

cv2.findChessboardCorners()
cv2.drawChessboardCorners()

cv2.findCirclesGrid()
cv2.findContours()
cv2.drawContours()

orb = cv2.ORB_create()
orb.detectAndCompute()
cv2.drawKeypoints()
bf = cv2.BFMatcher()
bf.match()
cv2.drawMatches()

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
flann = cv2.FlannBasedMatcher()
matches = bf.knnMatch()
matches = flann.knnMatch()
cv2.drawMatchesKnn()

cv2.distanceTransform()
cv2.subtract()
cv2.connectedComponents()

cv2.watershed()

cv2.namedWindow()
cv2.setMouseCallback()
cv2.destroyAllWindows()
cv2.EVENT_LBUTTONDOWN

face_cascade = cv2.CascadeClassifier()
face_rectangles = face_cascade.detectMultiScale()

cv2.calcOpticalFlowPyrLK()
cv2.calcOpticalFlowFarneback()

cv2.cartToPolar()
cv2.normalize()

dst = cv2.calcBackProject()
cv2.meanShift(dst, ...)
cv2.CamShift(dst, ...)

cv2.boxPoints()
cv2.polylines()

cv2.selectROI()

tracker = cv2.TrackerBoosting_create()
tracker = cv2.TrackerMIL_create()
tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerTLD_create()
tracker = cv2.TrackerMedianFlow_create()
tracker.init(frame, roi)
success, roi = tracker.update(frame)

cv2.goodFeaturesToTrack(...)
cv2.calcOpticalFlowPyrLK(...)

```

## Context: Image Representation

```python
# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Load image + access channels
image = cv2.imread('./image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_image)
red_channel = image[:,:,0]
green_channel = image[:,:,1]
blue_channel = image[:,:,2]
image.shape
img_resized = cv2.resize(image, (1100,600))

# Display
plt.figure(figsize=(20,10))
plt.imshow(gray_image, cmap='gray')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.imshow(red_channel, cmap='gray')
ax2.imshow(green_channel, cmap='gray')
ax3.imshow(blue_channel, cmap='gray')

# RGB Thresholding, Masking
lower_blue = np.array([0,0,220]) 
upper_blue = np.array([220,220,255])
mask = cv2.inRange(image, lower_blue, upper_blue)
masked_image = np.copy(image)
masked_image[mask != 0] = [0, 0, 0]
croped_masked_image = masked_image[0:514, 0:816]

# Convert (W, H, C) to (pixel, C)
pixel_vals = image.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)

# Compute color histograms
color = ('r','g','b')
for i,col in enumerate(color):
	hist = cv2.calcHist([image],channels=[i],mask=None,histSize=[256],ranges=[0,256])
	plt.plot(hist,col)

# Color maps
hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
sum_brightness = np.sum(hsv[:,:,2])
num_pixels = hsv.shape[0]*hsv.shape[1]
avg_brightness = sum_brightness / num_pixels

plt.hist(brightness_day, alpha = 0.5)
plt.hist(brightness_night, alpha = 0.5)
```

## Context: Filters, Fourier, Canny, Hough, Haar

```python

# FFT
def ft_image(norm_image):
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20*np.log(np.abs(fshift)) 
    return frequency_tx

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
norm_image = gray_image/255.0 # norm_image = gray_image.astype("float32")/255
fourier_image = ft_image(norm_image)

plt.imshow(fourier_image)
# all 4 quadrants symmetrical
# x,y: frequencies in x & y; pixel brightness: amplitude of that frequency pair
# center bright = solid colors, less texture, plane
# horizontal white lines = vertical structures, eg, persons
# vertical white lines = horizontal structures

# Lowpass filter (remove noise) -> Highpass filter (highlight features - edges) -> Threshold
gray_blur = cv2.GaussianBlur(gray_image, (9, 9), 0)
sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])
sobel_image_y = cv2.filter2D(gray_blur, -1, sobel_y)
retval, binary_image = cv2.threshold(sobel_image_y, 50, 255, cv2.THRESH_BINARY)

# Canny edges: lower & upper thresholds taken by observing grayvalue range & distribution
max_gray = np.amax(gray_image)
min_gray = np.amin(gray_image)
gray_hist = cv2.calcHist([gray_image],channels=[0],mask=None,histSize=[256],ranges=[0,256])
plt.plot(gray_hist)
lower = 60 # start with 
upper = 180 # 3x lower
edges = cv2.Canny(gray_image, lower, upper)

# Hough lines
edges = cv2.Canny(gray_image, low_threshold, high_threshold)
rho = 1
theta = np.pi/180
threshold = 60
min_line_length = 50
max_line_gap = 5
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
	                    min_line_length, max_line_gap)
line_image = np.copy(image)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# Hough circles
circles_im = np.copy(image)
circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 
                           minDist=40,
                           param1=70,
                           param2=11,
                           minRadius=25,
                           maxRadius=30)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(circles_im,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(circles_im,(i[0],i[1]),2,(0,0,255),3)
    plt.imshow(circles_im)
print('Circles shape: ', circles.shape)
plt.imshow(line_image)

# Haar Cascade Face Detector
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, 4, 6)
img_with_detections = np.copy(image)
for (x,y,w,h) in faces:
    cv2.rectangle(img_with_detections,(x,y),(x+w,y+h),(255,0,0),5)  
```

## Context: Harris, Contours, K-Means

```python
# Harris
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst,None)

# Contours: Detection
retval, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_image = np.copy(image)
contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3)

# Contours: Many features can be obtained from them!
(x,y), (MA,ma), angle = cv2.fitEllipse(cnt)
x,y,w,h = cv2.boundingRect(cnt)
M = cv2.moments(cnt)
m00 = M['m00']
m10 = M['m10']
area = cv2.contourArea(cnt)
...

# K-Means: Detect 3 most representative color groups (segment)
pixel_vals = image.reshape((-1,3)) # 2D array of pixels and 3 color values (RGB)
pixel_vals = np.float32(pixel_vals)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
k = 3 # 3 groups of colors
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# K-Means: Show segmented image
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])
plt.imshow(segmented_image)
```

## Context: Pyramids, ORB, HOG

```python
# Image pyramids
level_1 = cv2.pyrDown(image)
f, (ax1,ax2) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('original')
ax1.imshow(image)
ax2.imshow(level_1)
ax2.set_xlim([0, image.shape[1]])
ax2.set_ylim([image.shape[0], 0])

# ORB: matching of keypoints
orb = cv2.ORB_create(5000, 2.0)
keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)
query_img_keyp = copy.copy(query_image)
cv2.drawKeypoints(query_image, keypoints_query, query_img_keyp, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(descriptors_train, descriptors_query)
matches = sorted(matches, key = lambda x : x.distance)
result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:85], query_gray, flags = 2)
plt.imshow(result)
print(len(matches))

# HOG: train classificators (eg, SVM) with histograms of gradients in image cells
cell_size = (6, 6)
num_cells_per_block = (2, 2)
block_size = (num_cells_per_block[0] * cell_size[0],
              num_cells_per_block[1] * cell_size[1])
x_cells = gray_image.shape[1] // cell_size[0]
y_cells = gray_image.shape[0] // cell_size[1]
h_stride = 1
v_stride = 1
block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)
num_bins = 9        
win_size = (x_cells * cell_size[0] , y_cells * cell_size[1])
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
hog_descriptor = hog.compute(gray_image)
```

## Context: Optical Flow

```python
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

## Utilities: Re-Usable Snippets

### `resize_images()`

```python
def resize_images(image_paths, output_folder, min_size, keep_aspect_ratio=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_path in tqdm(image_paths, desc="Resizing images"):
        try:
            # Open image using PIL
            img = Image.open(img_path)
            # Convert to RGB
            img = img.convert("RGB")

            # In PIL w & h are transposed as compared to OpenCV
            w, h = img.size
            if keep_aspect_ratio:
                if h < w:
                    new_h, new_w = min_size, int(w * min_size / h)
                else:
                    new_h, new_w = int(h * min_size / w), min_size
            else:
                new_h, new_w = min_size, min_size

            # Resize the image
            #img = img.resize((new_w, new_h), Image.ANTIALIAS)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Save the image in the output folder with the same name
            output_path = os.path.join(output_folder, os.path.basename(img_path))
            img.save(output_path)
        except Exception as e:
            print(f"Failed to process {img_path}. Reason: {e}")

    print("Finished resizing images.")
```