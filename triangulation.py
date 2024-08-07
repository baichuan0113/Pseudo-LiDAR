import sys
import cv2
import numpy as np
import time

def find_depth(x_right, x_left, frame_right, frame_left, baseline, f, alpha):
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape
    f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
    disparity = x_left - x_right
    if disparity == 0:
        print("no disparity")
        return 0
    zDepth = (baseline * f_pixel) / disparity

    return abs(zDepth)

