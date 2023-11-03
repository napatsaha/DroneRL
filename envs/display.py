# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:51:49 2023

@author: napat
"""
from typing import List, Union

import cv2
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

from .geometry import Circle, Point, LineSegment, Canvas


class Mover(Circle):
    """
    A Movable object, used to subclass Predator and Prey.

    Inherited from Circle object for its colliding properties, but
    the display is a rectangle.
    """

    icon: np.ndarray
    name: str

    def __init__(self,
                 canvas_size: List[int],
                 icon_shape = (32,32),
                 image = None,
                 obstacle_list = None):

        # self.x = 0
        # self.y = 0

        if obstacle_list is None:
            self.obstacle_list = []
        else:
            self.obstacle_list = obstacle_list

        self.icon_w = icon_shape[1]
        self.icon_h = icon_shape[0]
        if image is not None:
            self.icon = cv2.imread(image, 0) / 255
            self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        else:
            self.icon = np.zeros((self.icon_h, self.icon_w))

        # Radius = Distance from center to corner of rectangle
        radius = ((self.icon_h / 2) **2 + (self.icon_w / 2)**2)**0.5
        
        self.canvas_size = canvas_size
        self.x_min = 0
        self.x_max = canvas_size[1]
        self.y_min = 0
        self.y_max = canvas_size[0]

        super().__init__(self.x_max/2, self.y_max/2, radius)
        
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

    def clamp_position_by_obstacle(self, obstacle: LineSegment):
        """Clamps position of circle with an obstable (line), to prevent
        passing through obstacle when moving."""
        G = self.closest_position_to_line(obstacle)

        if G is None:
            return
        else:
            sign_x, sign_y = self.direction_from(obstacle)

            clamp_x = min if sign_x < 0 else max
            clamp_y = min if sign_y < 0 else max

            self.x = clamp_x(self.x, G.x)
            self.y = clamp_y(self.y, G.y)
        
    def clamp_position(self):
        # Clamp by obstacle
        if len(self.obstacle_list) > 0:
            # Find nearest in-sight obstacle
            dist_list = [self.distance_to_line(obs) for obs in self.obstacle_list]
            idx = np.argmin(dist_list)
            obstacle = self.obstacle_list[idx]
            # Clamp only to nearest effective obstacle
            self.clamp_position_by_obstacle(obstacle)

        # for obs in self.obstacle_list:
        #     self.clamp_position_by_obstacle(obs)

        # Clamp by canvas edge
        self.x = self.clamp(self.x, round(self.x_min + self.icon_w/2), round(self.x_max - self.icon_w/2))
        self.y = self.clamp(self.y, round(self.y_min + self.icon_h/2), round(self.y_max - self.icon_h/2))
        
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def reset_position(self):
        pass

    def randomise_position(self):
        pass

    def add_obstacle(self, obstacle: Union[list, LineSegment]):
        if isinstance(obstacle, list):
            self.obstacle_list.extend(obstacle)
        else:
            self.obstacle_list.append(obstacle)

    def draw_on(self, canvas: Canvas):
        shapeX, shapeY = self.icon.shape
        canvas.canvas[
            round(self.x - 0.5*shapeX):round(self.x + 0.5*shapeX),
            round(self.y - 0.5*shapeY):round(self.y + 0.5*shapeY)
               ] = self.icon
        return canvas.canvas


class Predator(Mover):
    def __init__(self, canvas_size,
                 speed: float = 5,
                 icon_size = (32,32),
                 image = "drone2.png", **kwargs):
        super(Predator, self).__init__(canvas_size, icon_size, image=image, **kwargs)

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

    def sample_action(self) -> int:
        return self.action_space.sample()

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

    def sample_action(self) -> int:
        return self.action_space.sample()


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
        """
        if x is None: x = self.x_max / 4
        if y is None: y = self.y_max / 4

        self.set_position(x, y)