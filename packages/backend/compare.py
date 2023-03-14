import cv2
import numpy as np
from scipy.spatial.distance import cdist
import mantel

from contour import get_contours

SHAPE_DIFF_MAX = 8
OBJECT_DIFF_WEIGHT = 0.05
SHAPE_WEIGHT = 0.4
IMAGE_WEIGHT = 0.4
POSITION_WEIGHT = 0.2

def _center_position(c, img):
    size = img.shape
    m = cv2.moments(c)
    return (m["m10"] / m["m00"] / size[1], m["m01"] / m["m00"] / size[0])

def _contour_hist(c, img):
    left, top, width, height = cv2.boundingRect(c)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [c], (255))
    masked = cv2.bitwise_and(img, img, mask=mask)
    hsv = cv2.cvtColor(masked[top:top+height, left:left+width], cv2.COLOR_BGR2HSV)
    # showimage(hsv)
    hist = cv2.calcHist([hsv], [0, 1], None, [180,256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

def compare_image(img1, img2):
    if isinstance(img1, str):
        img1 = cv2.imread(img1)
    if isinstance(img2, str):
        img2 = cv2.imread(img2)

    r1 = get_contours(img1)
    r2 = get_contours(img2)

    # showimage(r1.image)
    # showimage(r2.image)
    # print((len(r1.contours), len(r2.contours)))


    r1.contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    r2pos = np.array([_center_position(c, r2.image) for c in r2.contours])
    r1pos = np.array([_center_position(c, r1.image) for c in r1.contours][:len(r2pos)])
    r2index = np.array(range(len(r2pos)))

    for i, pos in enumerate(r1pos):
        dists = np.sum((r2pos[i:] - pos) ** 2, axis=1)
        # print(r2pos[i:])
        min_index = i + np.argmin(dists)
        # Swap min_index to i-th row
        r2pos[[i, min_index]] = r2pos[[min_index, i]]
        r2index[[i, min_index]] = r2index[[min_index, i]]
        

    dist1 = cdist(r1pos, r1pos)
    dist2 = cdist(r2pos, r2pos)
    position_similarity = mantel.test(dist1, dist2, 0).r

    shape_differences = []
    image_similarities = []

    for i1, i2 in enumerate(r2index):
        # print(i1, i2)
        d = cv2.matchShapes(r1.contours[i1], r2.contours[i2], cv2.CONTOURS_MATCH_I1, 0.0)
        shape_differences.append(d)
        h1 = _contour_hist(r1.contours[i1], r1.image)
        h2 = _contour_hist(r2.contours[i2], r2.image)
        s = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        image_similarities.append(s)

    object_number_diff = abs(len(r1.contours) - len(r2.contours))
    r1area = np.array([cv2.contourArea(c) for c in r1.contours])

    shape_similarities = [1 - (d / SHAPE_DIFF_MAX) for d in shape_differences]
    shape_similarity = np.dot(shape_similarities, r1area) / np.sum(r1area)

    image_similarity = np.dot(image_similarities, r1area) / np.sum(r1area)

    print("Shape similarity: ", shape_similarity)
    print("Image similarity: ", image_similarity)
    print("Position similarity: ", position_similarity)
    print("Object number difference: ", object_number_diff)

    total_result = (1 - OBJECT_DIFF_WEIGHT * object_number_diff) * (SHAPE_WEIGHT * shape_similarity + IMAGE_WEIGHT * image_similarity + POSITION_WEIGHT * position_similarity)
    print("Total result: ", total_result)
    return total_result


compare_image("origin.png", "weird.png")