"""
Test __contains__ of Line class
"""

from envs.geometry import Circle, Line, Point, Canvas, generate_random_line, closest_point_on_line
import numpy as np
import matplotlib.pyplot as plt

W = 100
canvas = np.empty((W,W), dtype=np.int8)

# line = Line(Point(1,1), Point(7,7))
line = generate_random_line((W/10,W/10))

for i in range(W):
    for j in range(W):
        p = Point(i*0.1,j*0.1)
        col = int(p in line)
        canvas[i,j] = col

print(repr(line))

canvas = np.transpose(np.flip(canvas, 1))
plt.imshow(canvas)
plt.show()