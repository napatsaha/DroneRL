from envs.geometry import Canvas, create_polygon, Point, LineSegment, \
    generate_random_line, convert_line_to_box, distance_point_to_point
import numpy as np

W = 500
canvas = Canvas(W, W)
canvas.clear()

box = create_polygon(Point(100, 400),
                     Point(200, 400), Point(200,300),
                     Point(300, 300), Point(300, 400),
                     Point(400, 400),
                     Point(400, 100), Point(100, 100))

## Manual Method
# w = 10
# l = 100
# angle = np.pi * 7/8
# p1 = Point(200, 200)
# p2 = p1.move_point(angle, l)
# angle -= np.pi/2
# p3 = p2.move_point(angle, w)
# angle -= np.pi/2
# p4 = p3.move_point(angle, l)
# line = create_polygon(p1, p2, p3, p4)

lines = []
for i in range(5):
    orig_line = generate_random_line((W,W))
    # orig_line = LineSegment(Point(220, 220), Point(110,110))
    # orig_line = LineSegment(Point(110, 110), Point(220,220))
    line = convert_line_to_box(orig_line, 0.02*W)
    lines.append(line)

canvas.clear()
canvas.draw(box)
[canvas.draw(line) for line in lines]
# canvas.draw(line2)
canvas.show()

canvas.close()