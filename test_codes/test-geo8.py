"""
Test Circle intersection with Line
"""
from envs.geometry import InfLine, Circle, Canvas
import matplotlib.pyplot as plt

canvas = Canvas(100, 100)
canvas.clear()

circle = Circle(50,50,10)
line = InfLine(-1,0,55.0)

canvas.draw(circle)
canvas.draw(line)

plt.imshow(canvas.canvas)
plt.show()

print(circle.intersect(line))