import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

from perspective_transform import *
from lesson_functions import *

class CarFrame:
    def __init__(self,svc_file):
        dist_pickle = pickle.load( open(svc_file, "rb" ) )
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle["scaler"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.hist_bins = dist_pickle["hist_bins"]
        self.spatial_feat = dist_pickle['spatial_feat']
        self.hist_feat = dist_pickle['hist_feat']
        self.hog_feat = dist_pickle['hog_feat']
        self.color_space = dist_pickle['color_space']
        self.hog_channel = dist_pickle['hog_channel']
        self.framenumber = 0
        self.buffer_length = 20 #
        self.test_fail_array = []
        self.recent_frames_bbox = []
        self.best_bbox_list = []
        self.cars_found = 0
        self.miss_count = 0
        self.hot_windows = []
        self.hl_img1 = []
        self.hl_bbox1 = []
        self.heatmap1 = []
        self.hl_img2 = []
        self.hl_bbox2 = []
        self.heatmap2 = []
        self.avg_bbox = []
        self.draw_img = []

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def heatmap_labels(image,bbox_list,threshold,plot_img=1):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat,bbox_list)
    heat = apply_threshold(heat,threshold)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    #print('labels = ',labels)
    draw_img,labeled_bboxes = draw_labeled_bboxes(np.copy(image), labels,
                                                  plot_img)
    return draw_img,labeled_bboxes,heatmap,labels[1]


def draw_labeled_bboxes(img, labels,plot_img=1):
    # Iterate through all detected cars
    labeled_bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        labeled_bboxes.append(bbox)
        # Draw the box on the image
        if plot_img:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img,labeled_bboxes

def non_max_suppression_fast(boxes, overlapThresh):
    '''
    # Malisiewicz et al.
    Code to perform non maximum suppression for bounding boxes. Obtained from
    web at
    http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Was not able to get it to work as well as heatmap method, so not used in
    final code.
    '''
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    ################# add conversion code for lecture format of bboxes
    ## convert to numpy
    # make copy
    boxes = np.array(boxes)
    rboxes = np.copy(boxes) # save for later
    # flatten it
    boxes = boxes.flatten()
    # group into four tupples
    it = iter(boxes)
    boxes = list(zip(it,it,it,it))
    # convert back to numpy
    boxes = np.array(boxes)
    ############### end conversion code ##############
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    #return boxes[pick].astype("int")
    ############ more conversion back to lecture format
    nboxes = boxes[pick].astype("int")
    nboxes = nboxes.flatten()
    it = iter(nboxes)
    nboxes = list(zip(it,it))
    it = iter(nboxes)
    nboxes = list(zip(it,it))
    # rboxes = rboxes[pick].astype('int')
    # rboxes = rboxes.tolist()
    return nboxes

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    From class lecture, not used in final code.
    '''
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):
    '''
    Modified from class lecture code, not used in final code..
    '''

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1],
                              window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def find_cars(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell,
               cell_per_block, spatial_size, hist_bins,color_space):
    '''
    Implements hog subsampling which improves speed over search_windows()
    method. Adjust threshold for decisions is hard coded to 1.0 for this data.
    Would need to adjust this for different data and changes to number of
    scales used. Modified from lecture to include multiple scales in one call
    and color_space options. Two bug fixes and one found (marked in code).
    Inputs:
        img = image to be processed
        ystart,ystop = limits of y axis to search over
        scales = list of scale factors to compute hog over
        svc = linearSVC already trained
        X_scaler = StandardScaler().fit(X), needed to scale incoming data
        orient,pix_per_cell,cell_per_block = parameters for get_hog_features()
        spatial_size,hist_bins = parameters for bin_spatial() and color_hist()
        color_space = color space to use for processing, matches training
    Ouputs:
        bbox_list = list of bounding boxes with positive detections
    '''
    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255  # bad line from lecture code BUG FIX

    xstart = 0
    xstop = 1279  # BUG : not fixed. Scaling in x causes mirror image.
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch)


    test_decision_func = []
    bbox_list = []

    orig_tosearch = np.copy(ctrans_tosearch)

    for scale in scales:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale),
                                      np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        #nblocks_per_window = (window // pix_per_cell)-1 # bug found, replace
        # next line replaces above line reported as BUG from another student
        nblocks_per_window = window // pix_per_cell - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        #print('nxsteps = ',nxsteps,'nysteps = ',nysteps)

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                # subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window,
                #                      xleft:xleft+window], (64,64))
                #
                # ### Get color features if required
                #
                # spatial_features = bin_spatial(subimg, size=spatial_size)
                # hist_features = color_hist(subimg, nbins=hist_bins)
                #
                # #####  Scale features and make a prediction
                # test_features = X_scaler.transform(np.hstack((spatial_features,
                #                      hist_features, hog_features)).reshape(1, -1))

                test_features = X_scaler.transform(hog_features.reshape(1, -1))
                #test_prediction = svc.predict(test_features)
                #test_decision_func.append((test_prediction,svc.decision_function(test_features)))
                test_prediction = 0
                if svc.decision_function(test_features) > 1.0:
                     test_prediction = 1
                #test_prediction = 1
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    bot_left = (xbox_left, ytop_draw+ystart)
                    top_right = (xbox_left+win_draw,ytop_draw+win_draw+ystart)
                    bbox_list.append((bot_left,top_right))
                    #cv2.rectangle(draw_img, bot_left, top_right, (0,255,0),6)

    return draw_img,test_decision_func,bbox_list

def detect_car(image):
    '''
    Process pipeline for object detection of cars. Implemented both hog
    subsampling method (use_window = 0) and slower window search method. Needed
    both to debug the hog subsampling method.
    '''
    draw_image = np.copy(image)
    use_window=0
    if use_window:
        find_thresh=1
        keep_thresh = 1
        # compute windows only once at beginning
        y_start_stop = [400, 480]
        x_start_stop = [200,1080]
        windows1 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                            xy_window=(64, 64), xy_overlap=(0.75, 0.75))
        #print('len of window1 = ',len(windows1))
        y_start_stop = [380, 550]
        x_start_stop = [0,1279]
        windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                            xy_window=(96, 96), xy_overlap=(0.75, 0.75))
        #print('len of window2 = ',len(windows2))
        y_start_stop = [380, 656]
        x_start_stop = [0,1279]
        windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                            xy_window=(128, 128), xy_overlap=(0.75, 0.75))

        windows = windows1 + windows2 + windows3

    if use_window: # and (cf.framenumber % 5 == 0):
        # used to debug the hog_subsample method
        hot_windows = search_windows(image, windows, cf.svc, cf.X_scaler,
                                color_space=cf.color_space,
                                spatial_size=cf.spatial_size,
                                hist_bins=cf.hist_bins,
                                orient=cf.orient,
                                pix_per_cell=cf.pix_per_cell,
                                cell_per_block=cf.cell_per_block,
                                hog_channel=cf.hog_channel,
                                spatial_feat=cf.spatial_feat,
                                hist_feat=cf.hist_feat,
                                hog_feat=cf.hog_feat)

        draw_img,labeled_bboxes,heatmap,num_cars = heatmap_labels(image,hot_windows,
                                                         threshold=find_thresh,
                                                         plot_img=0)

        # TBD er ror checking here?
        #
        cf.recent_frames_bbox.append(labeled_bboxes)
        if len(cf.recent_frames_bbox) > cf.buffer_length:
            cf.recent_frames_bbox = cf.recent_frames_bbox[1:]

        # find recent frames with bounding boxes to sum over
        avg_bbox = []
        for box_list in cf.recent_frames_bbox:
            if box_list != 0:
                for box in box_list:
                    avg_bbox.append(box)

        # pass recent frames to get heatmap over multiple frames
        if avg_bbox:  # if not empty
            draw_img,labeled_bboxes,heatmap,num_cars = heatmap_labels(image,avg_bbox,
                                                threshold=keep_thresh,
                                                plot_img=1)
            cf.cars_found = num_cars
            cf.best_bbox_list = labeled_bboxes
            cf.last_car_found = cf.framenumber

    elif use_window==0 and cf.framenumber % 2 == 0:
        find_thresh=1
        keep_thresh=2
        scales = [1.0,1.5,2.0]
        ystart = 400
        ystop = 656
        fc_img,test_decision_func,hot_windows = find_cars(image, ystart, ystop,
                                scales, cf.svc, cf.X_scaler, cf.orient,
                                cf.pix_per_cell, cf.cell_per_block,
                                cf.spatial_size, cf.hist_bins,
                                cf.color_space)

        timg = draw_boxes(image,hot_windows,color=(0,255,0),thick=2)
        cf.hot_windows.append(timg)
        # get heatmap and collapse overlapping windows
        hl_img,labeled_bboxes,heatmap,num_cars = heatmap_labels(image,hot_windows,
                                                         threshold=find_thresh,
                                                         plot_img=1)
        cf.hl_img1.append(hl_img)
        cf.hl_bbox1.append(labeled_bboxes)
        cf.heatmap1.append(heatmap)
        # experimented with non-maximum-suppression instead of heatmap
        # did not work as well as heatmap
        # nboxes = non_max_suppression_fast(hot_windows, .3)
        # print('nboxes',nboxes)
        # labeled_bboxes=nboxes

        #cf.recent_hot_windows.append(hot_windows)
        cf.recent_frames_bbox.append(labeled_bboxes)
        if len(cf.recent_frames_bbox) > cf.buffer_length:
            cf.recent_frames_bbox = cf.recent_frames_bbox[1:]
            #cf.recnt_hot_windows = cf.recent_hot_windows[1:]

        # experimented with waiting to do heatmap until a few frames passed
        # did not work very well as boxes tended to overlap too much
        # hot_windows = [val for sublist in cf.recent_hot_windows for val in sublist]
        # hl_img,labeled_bboxes,heatmap,num_cars = heatmap_labels(image,hot_windows,
        #                                                  threshold=keep_hot_thresh,
        #                                                  plot_img=1)

        # find recent frames with bounding boxes to sum over
        avg_bbox = []
        for box_list in cf.recent_frames_bbox:
            if box_list != 0:
                for box in box_list:
                    avg_bbox.append(box)

        # pass recent frames to get heatmap over multiple frames
        if avg_bbox:  # if not empty
            draw_img,labeled_bboxes,heatmap,num_cars = heatmap_labels(image,avg_bbox,
                                                threshold=keep_thresh,
                                                plot_img=1)
            cf.cars_found = num_cars
            cf.best_bbox_list = labeled_bboxes
            cf.last_car_found = cf.framenumber
            cf.hl_img2.append(draw_img)
            cf.hl_bbox2.append(labeled_bboxes)
            cf.heatmap2.append(heatmap)
            timg = draw_boxes(image,avg_bbox,color=(0,255,0),thick=2)
            cf.avg_bbox.append(timg)
        else:
            cf.cars_found = 0 # no bounding boxes from recent frames

    if cf.cars_found:
        draw_img = draw_boxes(draw_image,cf.best_bbox_list)
        cf.miss_count=0
    else:
        cf.miss_count+=1
        if cf.framenumber > cf.buffer_length:
            if cf.miss_count < 3:
                #print('INSIDE MISS_COUNT')
                draw_img = draw_boxes(draw_image,cf.best_bbox_list)
            else:
                draw_img = draw_image
        else:
            draw_img = draw_image  # return unchanged image.
        #print('using original image...')

    #cf.test_fail_array.append(cf.cars_found)

    if cf.framenumber % 2 == 0:
        cf.draw_img.append(draw_img)
    cf.framenumber += 1
    return draw_img

if __name__ == '__main__':

    cf = CarFrame(svc_file='svc_pickle_orig.p')
    clip = VideoFileClip("project_video.mp4").subclip(37,42)
    #clip = VideoFileClip("project_video.mp4")
    project_clip = clip.fl_image(detect_car)
    # write output to video file
    project_clip.write_videofile('test_out_test.mp4',audio=False)

    pickle.dump([cf.hot_windows,cf.hl_img1,cf.hl_bbox1,cf.hl_bbox2,cf.hl_img2,
                cf.heatmap1,cf.heatmap2,cf.avg_bbox,cf.draw_img],open("test_save.p", "wb" ) )
