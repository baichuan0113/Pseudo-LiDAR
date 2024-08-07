import cv2 

#sift
sift = cv2.SIFT_create()

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


img1 = cv2.imread('1.jpg')  
img2 = cv2.imread('2.jpg') 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
# print(len(keypoints_1))
# print("------")
# print(len(keypoints_2))
for m in matches:
    # print(keypoints_1[m.queryIdx].pt)
    # print(keypoints_2[m.trainIdx].pt)
    print(m.distance)
    print("--------")

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches, img2, flags=2)

cv2.imshow('SIFT', img3)

cv2.waitKey(0)