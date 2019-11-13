# %%
import cv2 as cv
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os


# %%
# p = './PyTorch-YOLOv3-master/data/custom/images/0000038.png'
p = './RoiAndPos/T2019_131-pos_13132_25046.jpg'


def showCircles(p):
    img = cv.imread(p, 1)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gray = cv.medianBlur(gray, 5)
    print(np.max(gray))
    gray = np.uint8(np.exp(-(gray/255))*255)
    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, 1, minDist=10, param2=15, minRadius=2, maxRadius=30)
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(img, (i[0], i[1]), 0, (0, 0, 255), 1)
    print(len(circles[0, :]))
    cv.namedWindow('detected circles', 0)
    cv.imshow('detected circles', gray)
    cv.waitKey(0)
    cv.imshow('detected circles', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


showCircles(p)


# %%
def showImg(p):
    img = cv.imread(p, 1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # lower, upper = np.array([0, 0, 150]), np.array([180, 30, 255])
    lower, upper = np.array([0, 50, 0]), np.array([180, 255, 150])
    mask = cv.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    _, labels, stats, centroids = cv.connectedComponentsWithStats(mask)
    # stats 是bounding box的信息，N*5的矩阵，行对应每个label，五列分别为[x0, y0, width, height, area]
    # centroids 是每个域的质心坐标
    for stat in stats[1:]:
        x, y, w, h = stat[:4]
        img = cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.namedWindow('mask', 0)
    cv.imshow('mask', mask)
    cv.waitKey(0)
    cv.imshow('mask', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# showImg(p)
# %%
path = './RoiAndPos/'
dirs = os.listdir(path)
for dir in dirs:
    showImg(path+dir)


# %%
