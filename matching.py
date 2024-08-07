import siftFeatureMatching as sift
import cv2
from shapely.geometry import Point, Polygon
from roboflow import Roboflow
import numpy as np
from rtree import index
import triangulation as tri

def locate_points(polygon, points):
    res = []
    for point in points:
        p = Point(point)
        if p.within(polygon) and len(res) < 10:
            res.append(points.index(point))
    return res

def find_enclosing_bbox(keypoints):
    # Create an R-tree index
    idx = index.Index()
    # Insert bounding boxes of key points into the R-tree
    for i, point in enumerate(keypoints):
        x, y = point
        bbox = (x, y, x, y)  # Bounding box with the same point as min and max
        idx.insert(i, bbox)
    # Perform a range query to find the bounding box containing key points
    enclosing_bbox = idx.bounds

    return enclosing_bbox


def matching(prediction_l, matched_keypoints_l, matched_keypoints_r, frame_left, frame_right):
    print("matching!!!")
    for data in prediction_l:
        #center
        label = data['class']
        #connected = data['connected']
        x = data['x']
        y = data['y']
        width = data['width']
        height = data['height']

        # bounding box
        a = x - width / 2
        b = y - height / 2
        c = x + width / 2
        d = y + height / 2

        p1 = (a, b)
        p2 = (a, d)
        p3 = (c, b)
        p4 = (c, d)
        left_polygon = Polygon([p1, p2, p3, p4])

        pointsIndex_inside_left_bb = locate_points(left_polygon, matched_keypoints_l)

        sum_l = 0
        sum_r = 0

        points_inside_left_bb = []
        for p in pointsIndex_inside_left_bb:
            points_inside_left_bb.append(matched_keypoints_l[p])

        points_inside_right_bb = []
        for p in pointsIndex_inside_left_bb:
            points_inside_right_bb.append(matched_keypoints_r[p])

        if (len(points_inside_right_bb) == 0):
            print("no key points in the right")
            continue

        cv2.rectangle(frame_left, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        for point in points_inside_left_bb:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame_left, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        for point in points_inside_right_bb:
            x, y = int(point[0]), int(point[1])
            sum_r += x
            cv2.circle(frame_right, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
        right_center_x = sum_r / len(points_inside_right_bb)
        distance = tri.find_depth(right_center_x, x, frame_right, frame_left, baseline = 7, f = 2.88, alpha = 90)
        print(distance)
        cv2.putText(frame_left, label + f"{distance:.2f}", (int(a), int(b)), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)


