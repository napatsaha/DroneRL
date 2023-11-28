"""
Test LineSegment.extend().

Sequentially extend each line in an octagon, and render them separately.
"""
import numpy as np
from envs.geometry import Point, LineSegment, Circle, read_obstacle_config, Canvas

W = 1000
canvas = Canvas(W, W)
num_reps = 1

obs_list = read_obstacle_config("test_codes/obstacle-octagon.csv", canvas)

for i, obs in enumerate(obs_list):
    print("=====================")
    print("Obstacle", i)
    print(obs.point1, obs.point2)
    print("Angle:", np.degrees(obs.angle), end="\t\t")
    print("p1 < p2:", obs.point1 < obs.point2)
    canvas.clear()
    canvas.draw(obs)
    canvas.show()

    obs2 = obs.extends(0.1 * W)
    print(obs2.point1, obs2.point2)
    # print("Shorter" if obs2.length < obs.length else "Longer")
    print("Length", obs.length, "->", obs2.length)

    canvas.clear()
    canvas.draw(obs2)
    canvas.show()



canvas.close()
