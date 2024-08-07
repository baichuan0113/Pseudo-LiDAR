import sys
import cv2
import numpy as np
import time
import imutils

def find_circles(frame, mask):
    countours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countours = imutils.grab_contours(countours)
    center = None
    
    if (len(countours) > 0):
        c = max(countours, key = cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:

            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0,0,0), -1)
    return center

def detect_colored_point(frame, target_color, tolerance=10):
    # Convert the frame to BGR (assuming it's in BGR format)
    bgr_frame = frame

    # Define the color range for the target color with a tolerance
    lower_bound = np.array([c - tolerance for c in target_color], dtype=np.uint8)
    upper_bound = np.array([c + tolerance for c in target_color], dtype=np.uint8)

    # Create a mask by thresholding based on the color range
    mask = cv2.inRange(bgr_frame, lower_bound, upper_bound)

    # Find the coordinates of the first non-zero (white) pixel in the mask
    nonzero_coords = np.column_stack(np.where(mask > 0))

    center = None

    # Check if any non-zero pixels are found
    if len(nonzero_coords) > 0:
        # Use the first non-zero pixel as the detected point
        center = tuple(nonzero_coords[0])

        # Draw a circle at the detected point on the original frame
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    return center