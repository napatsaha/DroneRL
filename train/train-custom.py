# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:27:37 2023

@author: napat

Training using custom-written SAC algorithms
"""

import time, sys
from environment import DroneCatch

sys.path.append(r"C:\Users\napat\Unity\RobotArmRL")

from Training.gym_dsac import DSACAgent
# from Training.learning02 import Agent

from agent import DQNAgent

env = DroneCatch(resolution=(800, 800), frame_delay=5, reward_mult=10, dist_mult=0.1)
agent = DQNAgent(env)

log, t = agent.train(100)

env.frame_delay = 100
eval_results = agent.evaluate(delay=0, print_intermediate=True)
env.close()
