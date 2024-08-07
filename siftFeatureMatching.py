import cv2 
import matplotlib.pyplot as plt
import time

#sift
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
def get_sift(cap, cap1):
    while cap.isOpened() and cap1.isOpened():
        if cv2.waitKey(5) & 0xFF == 27:
            break
        # read images

        suc, img1 = cap.read()
        suc1, img2 = cap1.read()

        start = time.time()

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

        matches = bf.match(descriptors_1,descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)
        return matches, keypoints_1, keypoints_2
