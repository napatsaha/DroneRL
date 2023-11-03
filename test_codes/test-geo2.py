"""
Test if circle can detect collision with line.

When collided the screen freezes and waits for input,
as well as displaying previous action that led to collision.

Also test if drone agent can properly inherit from circle object.
"""

from envs.geometry import Circle, LineSegment, Point, Canvas, distance_line_point
from envs.display import Predator

W = 500
canvas = Canvas(W, W)

line = LineSegment(Point(0.3 * W, 0.6 * W), Point(0.7 * W, 0.2 * W))
# circle = Circle(Point(0.7*W, 0.9*W), 0.1*W)
# circle = Circle.from_coords(0.7 * W, 0.9 * W, 0.1 * W)
agent = Predator([W,W])

# agent.move_to_position(400, 400)

for ep in range(10):

    agent.randomise_position()

    # circle = BallCollider((W,W))

    while True:
        canvas.clear()
        a = agent.action_space.sample()
        agent.move(a)

        canvas.draw(line)

        canvas.draw(agent)

        canvas.show(10)

        if agent.detect_collision(line):
            print(f"Last action: {agent.convert_action(a)}")
            canvas.show(100)
            break

canvas.close()