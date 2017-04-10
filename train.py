import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler,RobustScaler
from skimage.feature import hog
import pickle
from lesson_functions import *
from sklearn.model_selection import train_test_split

# kitti_cars = glob.glob('vehicles/vehicles/KITTI_extracted/*.png',recursive=True)
# gti_far = glob.glob('vehicles/vehicles/GTI_Far/*.png',recursive=True)
# gti_far = sorted(gti_far)[::5] # get every fifth sample
# gti_left = glob.glob('vehicles/vehicles/GTI_Left/*.png',recursive=True)
# gti_left = sorted(gti_left)[::5] # get every fifth sample
# gti_mid = glob.glob('vehicles/vehicles/GTI_MiddleClose/*.png',recursive=True)
# gti_mid = sorted(gti_mid)[::5] # get every fifth sample
# gti_right = glob.glob('vehicles/vehicles/GTI_Right/*.png',recursive=True)
# gti_right = sorted(gti_right)[::5] # get every fifth sample
#
# cars = kitti_cars + gti_far + gti_left + gti_mid + gti_right

cars = glob.glob('vehicles/**/*.png',recursive=True)
notcars = glob.glob('non-vehicles/**/*.png',recursive=True)

# Reduce the sample size if needed
sample_size = None
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]
print('len cars = ',len(cars))
print('len notcars = ',len(notcars))

### TODO: Tweak these parameters and see how the results change.
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 100    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
########################## used in lecture example
color_space = 'YCrCb'
orient = 9
pix_per_cell= 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
##############################################
car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
#X_scaler = RobustScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
save_dict={}
save_dict['svc'] = svc
save_dict['scaler'] = X_scaler
save_dict['orient'] = orient
save_dict['pix_per_cell'] = pix_per_cell
save_dict['cell_per_block'] = cell_per_block
save_dict['spatial_size'] = spatial_size
save_dict['hist_bins'] = hist_bins
save_dict['spatial_feat'] = spatial_feat
save_dict['hist_feat'] = hist_feat
save_dict['hog_feat'] = hog_feat
save_dict['color_space'] = color_space
save_dict['hog_channel'] = hog_channel


#pickle.dump( save_dict,open("svc_pickle_orig.p", "wb" ) )
