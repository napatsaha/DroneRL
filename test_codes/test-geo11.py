from envs.geometry import InfLine, LineSegment, Point, Circle, closest_point_on_line, Canvas

W = 800
canvas = Canvas(W, W)

circle = Circle(425, 425, 10)


line1 = LineSegment(Point(400, 480), Point(500, 480))
line2 = LineSegment(Point(450, 400), Point(450, 500))
line1.intersect(line2)

print(closest_point_on_line(Point(circle.x, circle.y), line1))
print(circle.closest_position_to_line(line1))
print(closest_point_on_line(Point(circle.x, circle.y), line2))
print(circle.closest_position_to_line(line2))


iline1 = InfLine(0, 1, -480)
iline2 = InfLine(1, 0, -450)
P = iline1.intersect(iline2)

iline3 = iline1.find_perpendicular(P)

print(iline3 == iline2)
print(iline3 == line2)

canvas.clear()
canvas.draw([line2, line1, circle])
canvas.show(10)