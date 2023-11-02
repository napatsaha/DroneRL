from envs.geometry import Point, InfLine, Ray, quadrant, Line
import numpy as np


ray = Ray(np.pi * 1/2, Point(0,0))
# segment = Line(Point(-1, 5), Point(1, 5))
# segment = Line(Point(5,1), Point(5,-1))
segment = Line(Point(-1,5), Point(1,5))
# segment = Line(Point(-5,1), Point(-5,-1))
# segment = Line(Point(-1,-5), Point(1,-5))

ray.intersect(segment)