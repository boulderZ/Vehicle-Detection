
## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[figure1]: ./output_images/example_car_not_car.png
[figure2]: ./output_images/hog_example_Cr.png
[figure3]: ./output_images/example_output.jpg
[figure4]: ./output_images/pipeline_1.png
[figure5]: ./output_images/pipeline_2a.png
[figure6]: ./output_images/pipeline_final.png
[video1]: ./test_out_orig.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
#### Writeup / README

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used the skimage.feature hog function to extract hog features. The functions that implement all the hog extraction are in `lesson_functions.py`. The main function is `extract_features()` lines 46-101 and get_hog_features() lines 6-23.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][figure1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][figure2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I originally started with using hog features in combination with color histograms (`color_hist()` lines 34-42 of `lesson_functions.py` ) and spatial binning (`bin_spatial()` lines 26-30 of `lesson_functions.py`). I found after experimenting manually that the hog features alone were doing a good enough job of detecting cars and the other methods were not adding enough improvement to be worth the added computation time. I ended up using the following settings for the hog feature extraction:


| Setting        | Value   |
|:-------------:|:-------------:|
| color_space     | YCrCb        |
| orient     | 9      |
| pix_per_cell     | 8      |
| cell_per_block     |2        |
| hog_channel | ALL |


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in `train.py` using `LinearSVC` from `sklearn.svm`. I used all of the images from both the KITTI and GTI data sets. This resulted in over 8000 images from each classs (car, not_car). I experimented with limiting the GTI data set by using every fifth image to avoid overfitting, but it did not improve results over other methods that I was using to deal with false positives so I did not use it. I extracted features to train on using `extract_features()` in `lesson_functions.py`. I used `StandardScaler` from `sklearn.preprocessing` to scale and normalize features. The resulting classifier and scaler were saved to a pickle file after training completed. The test accuracy was 99.1% and it took 11.15 seconds to train.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started with the `search_windows()` method from the class notes (lines 227-259 in `window_search.py`). I experimented with using different scales and eventually settled on  64,96, and 128 square pixels and 75% overlap. I found this to work quite well but it was very very slow. Next I began implementing the hog subsample method `find_cars()` (lines 261-379 in `search_windows.py`) to try to get some speed up. I ran into problems right away with the results not matching the slower method and producing many more false positives. Eventually found that a scaling line that was needed for the example code in the lecture was causing the code to fail in my use case. Once that line was removed `img = img.astype(np.float32)/255`, everything started working. I then modified the `find_cars()` code to run multiple scales. I ended up using scales of 1.0, 1.5, and 2.0 as these closely matched the good results I achieved with the slower method. The overlap was 75% for the 1.0 case and higher for both of the other scales as I left the `cells_per_step =2` for all cases. That seemed to work best after experimenting.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on three scales using YCrCb 3-channel HOG features. I used the `decision_function()` option instead of `predict()` from the svc and thresholded that result to be > 1.0 for it to be a positive detection. This eliminated many false positives prior to using the heatmap method to group bounding boxes and eliminate frame to frame false positives. There are more examples in the next section.

![alt text][figure3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_out_orig.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

My pipeline has four main stages. In the first stage, I recorded the positions of positive detections in every other frame of the video from `find_cars()` using the hog subsampling method. I found processing every other frame worked well and gave a 2X speed up resulting in about 3.7 FPS. The first stage also has some rejection of false positives due to thresholding the decision function. The second stage creates a heatmap from the positive detections and then thresholds it with overlap of 1 to give the first estimate of vehicle locations and remove some false positives. The third stage buffers each of the stage 2 outputs for 20 cycles (40 frames) and only keeps the non-zero ones for processing in the final stage. The final stage takes the buffered bounding boxes from previous frames and creates another heatmap that is is then thresholded with overlap of 2 for the final output. The final stage removes more false positives. The code has logic to repeat the previous frame result if no bounding boxes were added. This is monitored and will reset if too many frames go by without an update. The logic for this is in the function `detect_car()`  in lines 459-540 of `window_search.py`  

Below shows example output from all the stages of the pipeline:

### Here are 3 cyles (6 frames in time) of first two stages :

![alt text][figure4]

### Here is the output of the final two stages showing buffer and heatmap
![alt text][figure5]

### Here the resulting bounding boxes after threshold applied
![alt text][figure6]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem was getting hog subsample method to work. This was critical because it sped up the pipeline dramatically making experiments easier to conduct. My current implementation did not use any hard negative mining during training and that might have helped with false positives. I felt that an improved rectangle grouping method would make the result much smoother. My current pipeline finds multiple cars on a single car during some of the frames. I experimented with non-maximum-suppression but found it was not working as well as the heatmap approach so abandoned it. Using more scales may have helped but also would have slowed down the pipeline. I believe that training on a birds eye view of the image and doing detection from that perspective may improve detection and reduce false positives. Another technique that likely would improve results would be a prediction filter on the bounding boxes.
