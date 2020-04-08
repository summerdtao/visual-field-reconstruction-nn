import cv2
import numpy as np


def load_image(filename):
    return cv2.imread(filename, 0)


def compare_img(base_img, compare_img):
    base_data = load_image(base_img)
    comp_data = load_image(compare_img)

    base_flatten = base_data.flatten()/255.0
    comp_flatten = comp_data.flatten()/255.0

    mse = (1- np.square(base_flatten - comp_flatten)).mean()
    return mse
