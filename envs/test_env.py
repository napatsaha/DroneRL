# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:30:57 2023

@author: napat

Simulate the predator to move around randomly while the prey moves in a circle.

Used for testing environment.
"""

import time
from envs.environment import DroneCatch
import matplotlib.pyplot as plt

manual = False
angle = 5
env = DroneCatch(resolution=800, frame_delay=10, reward_mult=10, dist_mult=0.1, 
                 manual_control=manual, random_predator=True, random_prey=True)

num_eps = 10

try:
    for i in range(num_eps):
        done = False
        env.reset()
        eps_reward, n_steps = 0,0
        while not done:
            
            action = env.render()
            if n_steps == 0: time.sleep(0.5)
            if not manual:
                action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            
            # print(obs, reward)
            
            eps_reward += reward
            n_steps += 1
            done = done | truncated
        
        status = "Done" if not truncated else "Truncated"
        print(f"Episode {i}\t Num Steps: {n_steps}\tReward: {eps_reward}\tStatus: {status}")

# for i in range(0, 360, angle):
#     action = env.action_space.sample()
#     env.step(action)
#     env.render()
#     # env.draw_canvas()
#     # plt.imshow(env.canvas, cmap=plt.cm.Greys_r)
#     # plt.show()
#     time.sleep(0.01)

finally:
    env.close()

# # Only for testing Key IDs
# env.render()
# for i in range(5):
#     key = cv2.waitKeyEx(0)
#     print(key)