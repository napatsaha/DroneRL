# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:49:46 2023

@author: napat
"""

import cv2
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

def draw_icon(canvas, icon, x, y):
    shapeX, shapeY = icon.shape
    canvas[round(x - 0.5*shapeX):round(x + 0.5*shapeX), 
           round(y - 0.5*shapeY):round(y + 0.5*shapeY)] = icon
    return canvas

def reset_canvas(size):
    return np.ones((size,size)) * 1

# plt.imshow(canvas)
# cv2.imshow("Drone", canvas)

canvasSize = 1000
droneProp = 0.05
canvas = reset_canvas(canvasSize)
drone_img = cv2.imread("drone2.png", 0) / 255
droneSize = int(droneProp*canvasSize)
drone_img = cv2.resize(drone_img, (droneSize, droneSize))


# pos = (int(0.7*canvasSize),int(0.1*canvasSize))
# canvas[pos[0]:pos[0]+drone_img.shape[0], pos[1]:pos[1]+drone_img.shape[1]] = drone_img

theta = 5
radius = 0.8*canvasSize / 2

angle = np.deg2rad(np.arange(0, 360, theta))
a = 0
while a != 13:
    for ag in angle:
        canvas = reset_canvas(canvasSize)
        y = radius * np.sin(ag) + canvasSize / 2
        x = radius * np.cos(ag) + canvasSize / 2
        draw_icon(canvas, drone_img, x, y)
        
    
        cv2.imshow("Drone", canvas)
        a = cv2.waitKey(30)
        if a == 13:
            break

cv2.destroyAllWindows()
