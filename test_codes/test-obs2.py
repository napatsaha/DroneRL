from envs.obstacles import Circle, Line, Point, Canvas, distance
from envs.display import Predator

W = 500
canvas = Canvas(W, W)

line = Line(Point(0.3 * W, 0.8 * W), Point(0.7 * W, 0.1 * W))
# circle = Circle(Point(0.7*W, 0.9*W), 0.1*W)
# circle = Circle.from_coords(0.7 * W, 0.9 * W, 0.1 * W)
agent = Predator([W,W])

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
            canvas.show(0)
            break

canvas.close()