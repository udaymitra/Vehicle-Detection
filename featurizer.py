from hog_computation import get_hog_features
from image_utils import *
import cv2
import numpy as np

# compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    return cv2.resize(img, size).ravel()

# compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def get_feature_vector(img, spatial=32, histbins=32,
                       orientations=8, pixels_per_cell=4, cells_per_block=2, transform_sqrt=True):
    hsv = convert_color_space(img, color_space='HSV')

    # color_features = bin_spatial(hsv, size=(spatial, spatial))
    # color_hist_features = color_hist(img, nbins=histbins)

    s_channel = get_channel(hsv, 1)
    hog_fv = get_hog_features(s_channel, orientations=orientations, pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block, transform_sqrt=transform_sqrt, visualise=False)

    # combine all features
    # fv = np.concatenate((color_features, color_hist_features, hog_fv))
    fv = hog_fv
    return fv