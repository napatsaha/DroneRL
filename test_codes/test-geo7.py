"""
Testing algorithm for raycasting using basic functions.
"""

from envs.geometry import Point, InfLine, Ray, quadrant, LineSegment, Canvas, generate_random_line, \
    distance_point_to_point, Circle
import numpy as np


W = 800
canvas = Canvas(W, W)

origin = Point(W/2,W/2)

circle = Circle(origin.x, origin.y, radius=W/40)
# ray = Ray(np.pi * 1/2, origin)
rays = [Ray(i * np.pi/4, origin) for i in range(8)]

n_lines = 5
lines = []

for _ in range(n_lines):
    line = generate_random_line(canvas.shape)
    lines.append(line)

crossed_rays = []
crossed_dist = []
for ray in rays:
    # crossed_lines = []
    distances = []
    ints_points = []
    for line in lines:
        P = ray.intersect(line)
        if P:
            ints_points.append(P)
            # crossed_lines.append(line)
            dist = distance_point_to_point(ray.origin, P)
            distances.append(dist)

    if len(ints_points) > 0:
        crossed_dist.append(min(distances))
        closest_point = ints_points[np.argmin(distances)]
        ray_trunced = LineSegment(ray.origin, closest_point)
        crossed_rays.append(ray_trunced)
    else:
        crossed_rays.append(ray)
        P = [ray.intersect(bound) for bound in canvas.boundaries if ray.intersect(bound)]
        # print(len(P))
        P = P[0]
        dist = distance_point_to_point(P, ray.origin)
        crossed_dist.append(dist)

# print(f"Number of intersected lines: {len(crossed_lines)}")
# print(f"Distances: {distances}")
print("Angle\tRaycast Distance")
for ray, dist in zip(rays, crossed_dist):
    print(f"{np.degrees(ray.angle)}\t{dist:.2f}")

canvas.draw(lines)
canvas.draw(circle)
canvas.show()

canvas.clear()
canvas.draw(lines)
canvas.draw(circle)
canvas.draw(crossed_rays)
canvas.show()

canvas.close()