import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
from roboflow import Roboflow
import calibration
import supervision as sv
import json
import time

import siftFeatureMatching as sift
import matching as match

import supervisely_lib as sly
#from supervisely_lib.api import SuperviselyAPI

# Functions
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri
#import calibration as calib
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 1
text_color = (255, 255, 255)  

rf = Roboflow(api_key="p8gvweVKXDaQ0AflqKDI")
project = rf.workspace().project("driving-qogkl")
model = project.version(2).model


B = 7             #Distance between the cameras [cm]
f = 2.88               #Camera lense's focal length [mm]

cap_right = cv2.VideoCapture('./resource/right_cut.mov')                 
cap_left =  cv2.VideoCapture('./resource/left_cut.mov')    

while(True):

    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    result_l = model.predict(frame_left, confidence=40, overlap=30).json()
    objects_l = result_l["predictions"]
    sift_res = sift.get_sift(cap_left, cap_right)
    matches = sift_res[0]
    keypoints_1 = sift_res[1]
    keypoints_2 = sift_res[2]

    matched_keypoints_l = [keypoints_1[match.queryIdx].pt for match in matches]
    matched_keypoints_r = [keypoints_2[match.trainIdx].pt for match in matches]

    match.matching(objects_l, matched_keypoints_l, matched_keypoints_l, frame_left, frame_right)
    frame_left_resized = cv2.resize(frame_left, (1000, 600))
    frame_right_resized = cv2.resize(frame_right, (1000, 600))


    cv2.imshow("frame left", frame_left_resized)
    cv2.imshow("frame right", frame_right_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
cap_left.release()
cap_right.release()
