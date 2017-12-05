[//]: # (Image References)
[image1]: ./output_images/data_visualization.png
[image2]: ./output_images/hog_visualization.png
[image3]: ./output_images/multiscale_detection_testimg1.jpg
[image4]: ./output_images/multiscale_detection_testimg2.jpg
[image5]: ./output_images/multiscale_detection_testimg3.jpg
[image6]: ./output_images/multiscale_detection_testimg4.jpg
[image7]: ./output_images/multiscale_detection_testimg5.jpg
[image8]: ./output_images/multiscale_detection_testimg6.jpg
[image9]: ./output_images/scale_1_5_detection.png


## Vehicle Detection and Tracking

<p align="center"> 
<img src="./misc/project_vid.gif" width = "720">
</p>

## Udacity Self Driving Car Engineer Nanodegree - Project 5

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## Files and Folders

* `vehicle-detect-pipeline.ipynb` : main software pipeline for this project 

* `functions.py`  : contains hog feature extract function

* `test_images` : images used to test the functions in this project

* `output_images` : images used to illustrate the steps taken to complete this project

* `output_videos` : final output video 

### Loading and Visualizing dataset
For this project I used the vehicle (labeled as cars) and non-vehicle (labeled as notcars) datasets provided by Udacity. Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

![alt text][image1]

### Extracting HOG features
The code for extracting HOG features from an image is defined by the function `get_hog_features` and is included in the file `functions.py`. The figure below shows a comparison of a car image and its associated histogram of oriented gradients, as well as the same for a non-car image.

![alt text][image2]

### Deciding the HOG parameters

I settled on my final choice of HOG parameters (shown below) based upon the performance of the SVM classifier produced using them. I considered not only the accuracy with which the classifier made predictions on the test dataset, but also the speed at which the classifier is able to make predictions. The YUV colorspace was found to produce less number of false positive detectinons. The training time took `2.4 seconds` to complete and the feature vector length using only HOG features was `1188` features

The final parameters chosen were YUV colorspace, 11 orientations, 16 pixels per cell, 2 cells per block, and ALL channels of the colorspace.

```python

color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
```

### Defining a function to extract features from a list of images
The function `extract_features` accepts a list of image paths, "cars" and "notcars", images and computes HOG parameters as well as color space conversion and produces a flattened array of HOG features for each image in the list.

### Training and testing the HOG Support Vector Classifier and the Color Histogram Support Vector Classifier
I trained a linear SVM with the defaultclassifier parameters and using HOG features alone (I did not use spatial intensity or channel intensity histogram features) and was able to achieve a test accuracy of `98.4 %`.

### Sliding Window Implementation
I used the `find_cars` function from the lesson materials and adapted them to use only HOG features. The method combines HOG feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the specified region of image (`y_start_stop = [400, 656]`) and then these full-image features are subsampled according to the size of the window and then fed to the classifier. 

The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

The image below shows the result of `find_cars` on one of the test images, using a single window size of **`1.5`**

![alt text][image9]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are six frames and their corresponding heatmaps with resulting bounding boxes. The following images is configured with search window scale of 1.0, 1.5, 2 and 3.5.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

---
### Video Implementation

Here's a [link to my video result](./output_videos/project_output.mp4)

### Adding Heatmaps and Bounding Boxes
The `add_heat` function increments the pixel value (referred to as "heat") of an all-black image the size of the original image at the location of each detection rectangle. Areas encompassed by more overlapping rectangles are thus assigned higher levels of heat

Then, a threshold is applied to the heatmap setting all pixels that don't exceed the threshold to zero.
The `scipy.ndimage.measurements.label()` function then collects spatially contiguous areas of the heatmap and assigns each a label.
And the final detection area is set to the extremities of each identified label using `draw_labeled_bboxes` function. The heatmaps can be seen in the above series of test_images.

The code for processing frames of video is `process_img`. The class `Vehicle_Detect` stores the detections (returned by find_cars) from the previous 15 frames of video using the prev_rects parameter. Rather than performing the heatmap/threshold/label steps for the current frame's detections, the detections for the past 15 frames are combined and added to the heatmap and the threshold for the heatmap is set to 1 + len(det.prev_rects)//2 (one more than half the number of rectangle sets contained in the history).

### Adding count of vehicles in frame
The `scipy.ndimage.measurements.label()` function returns two tuples, of which label[1] is the number of labels found. This count is displayed on the top left corner of the video. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The pipeline used in this project tends to do poorly when areas of the image darken by the presence of shadows. Classifying dark pixels as `cars`, creating false-positives. This issue could be resolved by adding more dark images to the `non-vehicle` dataset.
2. `xstart` and `xstop` is implemented as the left outer portion of the frame which is mostly useless and creates a lot of false-positives. These positions could be adjusted to reduce the number of false positives.
3. In the future I would like to try using deep-learning for vehicle recognition.

