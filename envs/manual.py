# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:53:05 2023

@author: napat
"""

import time
from environment import DroneCatch
import matplotlib.pyplot as plt

angle = 5
env = DroneCatch(resolution=(800, 800), frame_delay=10, reward_mult=10, dist_mult=0.1)

env.reset()
env.render()
action = 1
env.step(action)
env.close()
