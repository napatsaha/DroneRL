"""
Ray, InfLine, LineSegment and intersections

Test rays in a radial alignment from Point (0,0) to see if they intersect only
with line segments that crosses only each line.
"""

from envs.geometry import Point, InfLine, Ray, quadrant, LineSegment, Canvas
import numpy as np

# ray1 = Ray(np.pi/4, Point(0,0)) # y = x
# ray2 = Ray(np.pi/2, Point(0,0)) # x = 0 up
# ray3 = Ray(np.pi * 3/2, Point(0,0)) # x = 0 down
# ray4 = Ray(0, Point(0,0)) # y = 0

W = 800
canvas = Canvas(W, W)

origin = Point(W/2,W/2)

rays = [Ray(i * np.pi/4, origin) for i in range(8)]

# obs1 = InfLine.from_slope(-1, Point(0,5)) # y=-x+5
# obs2 = InfLine.from_slope(-1, Point(0,-5)) # y = -x-5

a = W/2.5
b = W/10
c = W/5

# Axis parallel lines
line1 = LineSegment(Point(a, b) + origin, Point(a, -b) + origin)
line2 = LineSegment(Point(-b, a) + origin, Point(b, a) + origin)
line3 = LineSegment(Point(-a, b) + origin, Point(-a, -b) + origin)
line4 = LineSegment(Point(-b, -a) + origin, Point(b, -a) + origin)

# Diagonal lines
lineq1 = LineSegment(Point(c, a) + origin, Point(a, c) + origin)
lineq2 = LineSegment(Point(-c, a) + origin, Point(-a, c) + origin)
lineq3 = LineSegment(Point(-c, -a) + origin, Point(-a, -c) + origin)
lineq4 = LineSegment(Point(c, -a) + origin, Point(a, -c) + origin)

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

canvas.clear()
canvas.draw(lines)
canvas.draw(rays)
canvas.show()
canvas.close()

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