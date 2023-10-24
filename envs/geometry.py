"""
Controls geometry elements.

Include geometry classes such as:
- Point
- Line
- Circle
as well as
- Canvas - for drawing objects on

Also handles distance calculations, collision detection, and
closest position for collision prevention between circle and line.

Mover class from display.py which controls Predator and Prey
inherits from Circle class.
"""


import math
import numpy as np
from abc import ABC, abstractmethod
from skimage import draw
import cv2

# from envs.display import Predator

class Geometry(ABC):

    @abstractmethod
    def draw_on(self, canvas):
        ...


class Point(Geometry):

    def __init__(self, x: float=0, y: float=0):
        self.x = x
        self.y = y

    def get_xy(self):
        return self.x, self.y

    def draw_on(self, canvas):
        pass

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def __repr__(self):
        return "Point{}".format(str(self))

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)


class AlgebraicLine(Geometry):
    pass

class Line(Geometry):

    def __init__(self, point1: Point, point2: Point):
        self.point1 = point1
        self.point2 = point2
        self.end_points = (self.point1, self.point2)
        self.length = distance_point_to_point(point1, point2)
        self.slope = slope_between_points(point1, point2)
        self.intercept = self.point1.y - self.slope * self.point1.x
        self.angle = np.arctan(self.slope)

    def substitute(self, x=None, y=None):
        if x is None and y is None:
            raise Exception("Both x and y cannot be missing.")
        elif y is None:
            y = self.slope * x + self.intercept
            return y
        elif x is None:
            x = (y - self.intercept) / self.slope
            return x

    # def __contains__(self, item: Point):
    #     if self.point1.x <= item.x <= self.point2.x and \
    #         self.point1.y

    def draw_on(self, canvas):
        xx, yy, val = draw.line_aa(
            int(self.point1.x), int(self.point1.y),
            int(self.point2.x), int(self.point2.y)
        )
        canvas[xx, yy] = val
        return canvas

    def __str__(self):
        return "y = {m}x + {b}".format(m=self.slope, b=self.intercept)


class Circle(Geometry):
    def __init__(self, center: Point, radius):
        self.x = center.x
        self.y = center.y
        self.center = center
        self.radius = int(radius)

    @classmethod
    def from_coords(cls, x, y, radius):
        """Accepts x,y coordinates as separate parameters instead of
        point objects."""
        return cls(Point(x,y), radius)

    def draw_on(self, canvas):
        xx,yy,val = draw.circle_perimeter_aa(
            int(self.x), int(self.y), self.radius,
        shape=canvas.shape)
        canvas[xx,yy] = val
        return canvas

    def detect_collision(self, line: Line):
        center = Point(self.x, self.y)
        # First check if passed line segment ends
        # if any([distance_point_to_point(center, end_point) > line.length for end_point in (line.point1, line.point2)]):
        if self.is_clear_of_line(line):
            return False
        # Otherwise check if radius is less than distance
        else:
            d = distance_line_point(line, center)
            return d <= self.radius

    def closest_position_to_line(self, line: Line):
        if self.is_clear_of_line(line):
            return None

        # Determines whether circle is above or underneath the line
        sign_y = np.sign(self.y - line.substitute(x=self.x))
        sign_x = np.sign(self.x - line.substitute(y=self.y))

        # Closest point ON the line
        P = closest_point_on_line(Point(self.x, self.y), line)

        # Deviation from P to safe point G
        x = sign_x * self.radius * np.sin(np.abs(line.angle))
        y = sign_y * self.radius * np.cos(np.abs(line.angle))

        # Add deviation to point P to get G
        G = P + Point(x,y)

        return G

    def is_clear_of_line(self, line: Line):
        """Check whether circle is in a position where the closest perpendicular point lies
        outside line segment."""
        center = Point(self.x, self.y)
        dist = max(*(distance_point_to_point(center, end) for end in line.end_points))
        is_clear = (dist + self.radius) >= line.length
        return is_clear

class Canvas:

    def __init__(self, width: int, height: int = None,
                 name: str = "environment"):
        if height is None:
            height = width

        self.width = width
        self.height = height

        self.canvas = np.ones((self.height, self.width))
        self.name = name

    def draw(self, obj: Geometry):
        self.canvas = obj.draw_on(self.canvas)

    def show(self, frame_delay: int=0):
        # Transform [i,j] indexing to (x,y) coordinates
        cartesian = np.transpose(np.flip(self.canvas, 1))
        cv2.imshow(self.name, cartesian)
        key = cv2.waitKey(frame_delay)
        return key

    def clear(self):
        self.canvas = np.ones((self.height, self.width))

    def close(self):
        cv2.destroyWindow(self.name)

# Add collisions

def distance_line_point(line: Line, point: Point):
    x0, y0 = point.get_xy()
    x1, y1 = line.point1.get_xy()
    x2, y2 = line.point2.get_xy()

    numer = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))
    denom = math.sqrt((x2-x1)**2 + (y2-y1)**2)

    d = numer / denom
    return d

def distance_point_to_line(point: Point, line: Line):
    return distance_line_point(line, point)

def distance_point_to_point(point1: Point, point2: Point):
    """Euclidean distance between two points"""
    x1, y1 = point1.get_xy()
    x2, y2 = point2.get_xy()

    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d

def slope_between_points(point1: Point, point2: Point):
    return (point2.y - point1.y) / (point2.x - point1.x)


def closest_point_on_line(point: Point, line: Line):
    # Find perpendicular line
    perpendicular_slope = -1/(line.slope)
    perpendicular_intercept = point.y - perpendicular_slope * point.x # y=mx+b -> b=y-mx

    # Find intersecting point
    x = (perpendicular_intercept - line.intercept) / (line.slope - perpendicular_slope)
        # a1x + b1 = a2x + b2
        # x = (b2-b1) / (a1-a2)
    y = line.substitute(x=x)

    return Point(x, y)


if __name__ == "__main__":
    point = Point(1, 1)

    W = 500
    canvas = Canvas(W, W)

    line = Line(Point(0.3*W, 0.8*W), Point(0.7*W, 0.1*W))
    # circle = Circle(Point(0.7*W, 0.9*W), 0.1*W)
    circle = Circle.from_coords(0.5*W, 0.5*W, 0.1*W)

    print(circle.detect_collision(line))

    # circle = BallCollider((W,W))

    canvas.draw(line)
    canvas.draw(circle)

    canvas.show(0)

    canvas.close()
