import math
import numpy as np
from abc import ABC, abstractmethod
from skimage import draw
import cv2

from envs.display import Predator

class Geometry(ABC):

    @abstractmethod
    def draw_on(self, canvas):
        ...


class Point(Geometry):

    def __init__(self, x=0, y=0):
        self.x = int(x)
        self.y = int(y)

    def get_xy(self):
        return self.x, self.y

    def draw_on(self, canvas):
        pass


class Line(Geometry):

    def __init__(self, point1: Point, point2: Point):
        self.point1 = point1
        self.point2 = point2

    def draw_on(self, canvas):
        xx, yy, val = draw.line_aa(
            self.point1.x, self.point1.y,
            self.point2.x, self.point2.y
        )
        canvas[xx, yy] = val
        return canvas


class Circle(Geometry):
    def __init__(self, center: Point, radius):
        self.x = center.x
        self.y = center.y
        self.center = center
        self.radius = int(radius)

    def draw_on(self, canvas):
        xx,yy,val = draw.circle_perimeter_aa(
            self.x, self.y, self.radius,
        shape=canvas.shape)
        canvas[xx,yy] = val
        return canvas

    def detect_collision(self, line: Line):
        d = distance(self.center, line)
        return d <= self.radius


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
        cv2.waitKey(frame_delay)

    def close(self):
        cv2.destroyWindow(self.name)

# Add collisions

def distance(line: Line, point: Point):
    x0, y0 = point.get_xy()
    x1, y1 = line.point1.get_xy()
    x2, y2 = line.point2.get_xy()

    numer = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))
    denom = math.sqrt((x2-x1)**2 + (y2-y1)**2)

    d = numer / denom
    return d


if __name__ == "__main__":
    point = Point(1, 1)

    W = 500
    canvas = Canvas(W, W)

    line = Line(Point(0.3*W, 0.8*W), Point(0.7*W, 0.1*W))
    # circle = Circle(Point(0.7*W, 0.9*W), 0.1*W)

    circle = BallCollider((W,W))

    canvas.draw(line)
    canvas.draw(circle)

    canvas.show(0)
    canvas.close()
