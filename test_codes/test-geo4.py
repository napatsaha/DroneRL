"""
Test collision avoidance, making sure agent cannot pass through obstacle
when colliding.

Agent moves manually using arrow keys.
"""

from envs.geometry import Circle, Line, Point, Canvas, distance_line_point
from envs.display import Predator

key_dict = {
            32     : 0, # Space Arrow
            2490368: 1, # Up Arrow
            2424832: 2, # Left Arrow
            2621440: 3, # Down Arrow
            2555904: 4, # Right Arrow
            }

W = 500
canvas = Canvas(W, W)

line = Line(Point(0.3 * W, 0.6 * W), Point(0.7 * W, 0.2 * W))
# circle = Circle(Point(0.7*W, 0.9*W), 0.1*W)
# circle = Circle.from_coords(0.7 * W, 0.9 * W, 0.1 * W)
agent = Predator([W,W], obstacle_list=[line])

# agent.move_to_position(400, 400)

for ep in range(10):

    agent.randomise_position()

    # circle = BallCollider((W,W))

    for sp in range(100):
        canvas.clear()
        canvas.draw(line)
        canvas.draw(agent)

        key = canvas.show(manual=True)
        action = key_dict[key]
        agent.move(action)


canvas.close()