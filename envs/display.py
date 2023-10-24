# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:51:49 2023

@author: napat
"""
from typing import List

import cv2
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

from .geometry import Circle, Point


class Mover(Circle):

    icon: np.ndarray
    name: str

    def __init__(self, canvas_size: List[int], icon_shape = (32,32), image = None):

        # self.x = 0
        # self.y = 0

        self.icon_w = icon_shape[1]
        self.icon_h = icon_shape[0]
        if image is not None:
            self.icon = cv2.imread(image, 0) / 255
            self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        else:
            self.icon = np.zeros((self.icon_h, self.icon_w))

        radius = ((self.icon_h / 2) **2 + (self.icon_w / 2)**2)**0.5
        
        self.canvas_size = canvas_size
        self.x_min = 0
        self.x_max = canvas_size[1]
        self.y_min = 0
        self.y_max = canvas_size[0]

        super().__init__(Point(self.x_max/2, self.y_max/2), radius)
        
    def set_position(self, x, y):
        self.x = x
        self.y = y
        
        self.clamp_position()
        
    def get_position(self, normalise=False):
        scale = self.x_max if normalise else 1.0
        return np.array((self.x, self.y)) / scale
        
    def move_to_position(self, x, y):
        self.x += x
        self.y += y
        
        self.clamp_position()
        
    def clamp_position(self):
        self.x = self.clamp(self.x, round(self.x_min + self.icon_w/2), round(self.x_max - self.icon_w/2))
        self.y = self.clamp(self.y, round(self.y_min + self.icon_h/2), round(self.y_max - self.icon_h/2))
        
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def reset_position(self):
        pass

    def randomise_position(self):
        pass

    def draw_on(self, canvas):
        shapeX, shapeY = self.icon.shape
        canvas[
            round(self.x - 0.5*shapeX):round(self.x + 0.5*shapeX),
            round(self.y - 0.5*shapeY):round(self.y + 0.5*shapeY)
               ] = self.icon
        return canvas


class Predator(Mover):
    def __init__(self, canvas_size,
                 speed: float = 5,
                 icon_size = (32,32),
                 image = "drone2.png"):
        super(Predator, self).__init__(canvas_size, icon_size, image=image)

        self.name = "predator"
        self.speed = speed
        self.move_speed = round(self.speed * 0.01 * self.canvas_size[0])
        # self.icon = cv2.imread(image, 0) / 255
        # self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

        self.action_space = spaces.Discrete(5)
        
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

    def convert_action(self, action):
        """
        Converts scalar action into (x,y) directional movement.

        0: (0, 0)
        1: (0, 1) # Up
        2: (-1, 0) # left
        3: (0, -1) # Down
        4: (1, 0) # Right

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        """
        if action > 0:
            x = np.cos(action * np.pi / 2).astype('int')
            y = np.sin(action * np.pi / 2).astype('int')
        else:
            x, y = 0, 0
        return x, y

    def move(self, action):
        delta = np.array([*self.convert_action(action)]) * \
                self.move_speed
        self.move_to_position(*delta)

    def randomise_position(self):
        rand_pos = np.random.randint(self.canvas_size)
        self.reset_position(*rand_pos)

class AngularPrey(Mover):
    def __init__(self, canvas_size, angle_delta, radius,
                 icon_size = (32,32),
                 image = "drone.png"):
        super(AngularPrey, self).__init__(canvas_size, icon_size, image=image)
        self.name = "prey"
        # self.icon = cv2.imread(image, 0) / 255
        # self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

        self.action_space = spaces.Discrete(3)

        self.angle_delta = angle_delta
        self.radius = radius
        self.reset_position()
        
    def reset_position(self, default_angle: int=0):
        """
        Reset angle of AngularPrey on the circumference. Optionally, an angle can be specified.

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
            assert 0 <= action <= 2, "AngularPrey Action out of range: Only accept [0, 1, 2]"
            mapper = {0:0, 1:-1, 2:1}
            action = mapper[action]
            
        self.angle = (self.angle + action * self.angle_delta) % 360
        self.update_position()

    def move(self, action):
        self.move_in_circle(action)

    def randomise_position(self):
        angle = np.random.randint(360)
        self.reset_position(angle)


class CardinalPrey(Predator):
    def __init__(self, canvas_size, **kwargs):
        """
        A version of prey that allows cardinal direction movement instead of
        in a circle.

        Identical to predator except uses prey drone image, and defaults to
        spawning in upper left quadrant instead of in the centre.

        """
        super().__init__(canvas_size, image="drone.png", **kwargs)
        self.name = "prey"

    def reset_position(self, x: float = None, y: float = None) -> None:
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
        if x is None: x = self.x_max / 4
        if y is None: y = self.y_max / 4

        self.set_position(x, y)