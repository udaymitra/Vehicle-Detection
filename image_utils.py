import numpy as np
import cv2

def read_image_as_rgb(file_path):
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def convert_color_space(img, color_space = 'RGB'):
    transformed_image = img
    if color_space != 'RGB':
        if color_space == 'HSV':
            transformed_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            transformed_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            transformed_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            transformed_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            transformed_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return transformed_image

def get_channel(img, channel_number):
    assert channel_number >= 0 and channel_number < 3
    return img[:, :, channel_number]

def convert_color_space_and_get_channel(img, color_space = 'RGB', channel_number = 2):
    transformed_img = convert_color_space(img, color_space)
    return get_channel(transformed_img, channel_number)

def flip_image(img):
    return np.fliplr(img)