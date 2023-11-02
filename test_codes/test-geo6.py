"""
Ray, InfLine, Line and intersections
"""

from envs.geometry import Point, InfLine, Ray, quadrant, Line
import numpy as np

# ray1 = Ray(np.pi/4, Point(0,0)) # y = x
# ray2 = Ray(np.pi/2, Point(0,0)) # x = 0 up
# ray3 = Ray(np.pi * 3/2, Point(0,0)) # x = 0 down
# ray4 = Ray(0, Point(0,0)) # y = 0

rays = [Ray(i * np.pi/4, Point(0,0)) for i in range(8)]

# obs1 = InfLine.from_slope(-1, Point(0,5)) # y=-x+5
# obs2 = InfLine.from_slope(-1, Point(0,-5)) # y = -x-5
line1 = Line(Point(5,1), Point(5,-1))
line2 = Line(Point(-1,5), Point(1,5))
line3 = Line(Point(-5,1), Point(-5,-1))
line4 = Line(Point(-1,-5), Point(1,-5))

lineq1 = Line(Point(2,5), Point(5,2))
lineq2 = Line(Point(-2,5), Point(-5,2))
lineq3 = Line(Point(-2,-5), Point(-5,-2))
lineq4 = Line(Point(2,-5), Point(5,-2))

lines = [
    line1, lineq1,
    line2, lineq2,
    line3, lineq3,
    line4, lineq4
]

for line in lines:
    print(f"{str(line).ljust(25)}", end="\t")
    for ray in rays:
        angle = np.rad2deg(ray.angle)
        point = ray.intersect(line)
        print(f"{str(point).ljust(15)}", end="\t")
    print(end="\n")

# [ray.intersect(line1) for ray in rays]
# [ray.intersect(line2) for ray in rays]
# [ray.intersect(line3) for ray in rays]
# [ray.intersect(line4) for ray in rays]
#
# ray1.intersect(obs1)
# ray1.intersect(obs2)
#
# ray2.intersect(obs1)
# ray2.intersect(obs2)
#
# ray3.intersect(obs2)