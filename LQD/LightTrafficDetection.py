#!/usr/bin/env python
# coding: utf-8
# created by khoivotri@gmail.com
# Date: 11/11/2023
# Time: 10:00
#
import cv2
import numpy as np

img = cv2.imread("RedGreenYellow", cv2.COLOR_BGR2HSV)

red1 = cv2.inRange(img, (0,100,100), (10,255,255))
red2 = cv2.inRange(img, (160,100,100), (180,255,255))
green = cv2.inRange(img, (40,50,50), (90,255,255))
yellow = cv2.inRange(img, (15,150,150), (35,255,255))
red = cv2.add(red1, red2)

shape = img.shape

cir_red = cv2.HoughCircles(red, cv2.HOUGH_GRADIENT, 1, 80, 
                            param1 = 50, param2 = 10, minRadius = 0, maxRadius = 30)
cir_green = cv2.HoughCircles(green, cv2.HOUGH_GRADIENT, 1, 60,
                            param1 = 50, param2 = 10, minRadius = 0, maxRadius = 30)
cir_yellow = cv2.HoughCircles(yellow, cv2.HOUGH_GRADIENT, 1, 30,
                            param1 = 50, param2 = 10, minRadius = 0, maxRadius = 30)

bound = 4.0 / 10

if cir_red is not None:
        cir_red = np.uint16(np.around(cir_red))
        for i in cir_red[0, :]:
            if i[0] > shape[1] or i[1] > shape[0]or i[1] > shape[0] * bound: continue
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)

if cir_green is not None:
        cir_green = np.uint16(np.around(cir_green))
        for i in cir_green[0, :]:
            if i[0] > shape[1] or i[1] > shape[0]or i[1] > shape[0] * bound: continue
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)

if cir_yellow is not None:
        cir_yellow = np.uint16(np.around(cir_yellow))
        for i in cir_yellow[0, :]:
            if i[0] > shape[1] or i[1] > shape[0]or i[1] > shape[0] * bound: continue
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)

cv2.imshow("Detect", img)
cv2.waitKey(0)
cv2.destroyAllWindows()