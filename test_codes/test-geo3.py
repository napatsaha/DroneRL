from envs.obstacles import Point, Line, Circle, Canvas
import numpy as np
import cv2

W = 500
radius = 10

canvas = Canvas(W, W)

key = 0
while key >=0 and key not in (27,):

    canvas.clear()
    line = Line(*(Point(*np.random.randint(W, size=2)) for _ in range(2)))
    circle = Circle(Point(*np.random.randint(W, size=2)), radius)

    G = circle.closest_position_to_line(line)



    canvas.draw(line)
    canvas.draw(circle)
    if G is not None:
        circle2 = Circle(G, radius)
        canvas.draw(circle2)

    key = canvas.show(0)
    print(key)

canvas.close()