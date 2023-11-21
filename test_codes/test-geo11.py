from envs.geometry import InfLine, LineSegment, Point, Circle, closest_point_on_line, Canvas

W = 800
canvas = Canvas(W, W)

circle = Circle(425, 425, 10)


hline = LineSegment(Point(400, 480), Point(500, 480))
vline = LineSegment(Point(450, 400), Point(450, 500))
hline.intersect(vline)

# print(closest_point_on_line(Point(circle.x, circle.y), hline))
p1 = circle.closest_position_to_line(hline)
print(p1)
print(circle.direction_from(p1))

# print(closest_point_on_line(Point(circle.x, circle.y), vline))
p2 = circle.closest_position_to_line(vline)
print(p2)
print(circle.direction_from(p2))


iline1 = InfLine(0, 1, -480)
iline2 = InfLine(1, 0, -450)
P = iline1.intersect(iline2)

iline3 = iline1.find_perpendicular(P)

# print(iline3 == iline2)
# print(iline3 == vline)

canvas.clear()
canvas.draw([vline, hline, circle])
canvas.show(0)