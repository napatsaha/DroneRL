# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:51:49 2023

@author: napat
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Point(object):
    def __init__(self, canvas_size, icon_shape = (32,32)):
        
        self.x = 0
        self.y = 0
        self.icon_w = icon_shape[1]
        self.icon_h = icon_shape[0]
        
        self.canvas_size = canvas_size
        self.x_min = 0
        self.x_max = canvas_size[1]
        self.y_min = 0
        self.y_max = canvas_size[0]
        
    def set_position(self, x, y):
        self.x = x
        self.y = y
        
        self.clamp_position()
        
    def get_position(self):
        return np.array((self.x, self.y))
        
    def move(self, x, y):
        self.x += x
        self.y += y
        
        self.clamp_position()
        
    def clamp_position(self):
        self.x = self.clamp(self.x, round(self.x_min + self.icon_w/2), round(self.x_max - self.icon_w/2))
        self.y = self.clamp(self.y, round(self.y_min + self.icon_h/2), round(self.y_max - self.icon_h/2))
        
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
    
class Predator(Point):
    def __init__(self, canvas_size, icon_size = (32,32)):
        super(Predator, self).__init__(canvas_size, icon_size)
        
        self.icon = cv2.imread("drone2.png", 0) / 255
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
        self.reset_position()

    def reset_position(self, x: float=None, y: float=None) -> None:
        """
        Reset position of Predator. Optionally, (x,y) coordinates can be specified.

        Parameters
        ----------
        x : float, optional
            DESCRIPTION. The default is None.
        y : float, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None
            DESCRIPTION.

        """
        if x is None: x = self.x_max/2
        if y is None: y = self.y_max/2
        
        self.set_position(x, y)
    
class Prey(Point):
    def __init__(self, canvas_size, angle_delta, radius, icon_size = (32,32)):
        super(Prey, self).__init__(canvas_size, icon_size)
        self.icon = cv2.imread("drone.png", 0) / 255
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
        self.angle_delta = angle_delta
        self.radius = radius
        self.reset_position()
        
    def reset_position(self, default_angle: int=0):
        """
        Reset angle of Prey on the circumference. Optionally, an angle can be specified.

        Parameters
        ----------
        default_angle : int, optional
            Starting angle for the prey to spawn in. The default is 0.

        Returns
        -------
        None.

        """
        self.angle = default_angle
        self.update_position()        
        
    def update_position(self):
        """Converts current angle to x, y positions"""
        angle = np.deg2rad(self.angle)
        self.y = self.radius * np.sin(angle) + self.y_max / 2
        self.x = self.radius * np.cos(angle) + self.x_max / 2
        
    def move_in_circle(self, action: int = None):
        """Move position counter-clockwise by angle_delta each step."""
        if action is None:
            action = 1
        else:
            assert 0 <= action <= 2, "Prey Action out of range: Only accept [0, 1, 2]"
            mapper = {0:0, 1:-1, 2:1}
            action = mapper[action]
            
        self.angle = (self.angle + action * self.angle_delta) % 360
        self.update_position()