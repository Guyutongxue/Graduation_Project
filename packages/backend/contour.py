import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

BLUR = 21
CANNY_LOW = 15
CANNY_HIGH = 150
MIN_AREA = 0.0005
MAX_AREA = 0.95
# DILATE_ITER = 10
# ERODE_ITER = 10

ContourResult = namedtuple("ContourResult", ["image", "contours", "background_color"])

def average_color(img, mask):
    flat_mask = mask.flatten()
    flat_img = img.reshape(-1, img.shape[-1])
    masked = flat_img[flat_mask]
    return np.mean(masked, axis=0)

def get_contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, CANNY_LOW, CANNY_HIGH)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    len(contours)

    total_area = img.shape[0] * img.shape[1]
    max_area = total_area * MAX_AREA
    min_area = total_area * MIN_AREA


    contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    contour_rects = [cv2.boundingRect(c) for c in contours]
    left, top, right, bottom = (np.array(contour_rects) @ np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])).T
    left, top, right, bottom = (left.min(), top.min(), right.max(), bottom.max())

    # background mask
    mask = np.zeros(edges.shape, dtype = np.uint8)
    for c in contours:
        mask = cv2.fillConvexPoly(mask, c, (255))
    
    cropped = img[top:bottom, left:right]
    cropped_contours = [c - [left, top] for c in contours]

    background_color = average_color(img, mask)
    return ContourResult(cropped, cropped_contours, background_color)
