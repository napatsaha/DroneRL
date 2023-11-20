"""
Test Obstacle config file and reading
"""

import numpy as np
from envs.geometry import Canvas, read_obstacle_config

W = 800
canvas = Canvas(W, W)

file = "test_codes/obstacle.csv"
obstacles = read_obstacle_config(file, canvas)

canvas.clear()
canvas.draw(obstacles)
canvas.show()