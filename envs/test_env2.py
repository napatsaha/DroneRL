# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: napat

Test MultiDrone env
"""
import time
from envs import *
import numpy as np

env = MultiDrone(
    num_predators=2,
    num_preys=1,
    min_distance=0.2,
    reward_distance_strategy="individual-minimum",
    observation_distance_strategy="individual-all",
    use_relative_position=True,
    include_own_position=False,
    prey_move_speed=10,
    verbose=4
)

num_eps = 10
delay = 10

env.set_frame_delay(delay)

try:
    for i in range(num_eps):
        done = False
        env.reset()
        eps_reward, n_steps = 0, 0
        while not done:

            env.render()
            # if n_steps == 0: time.sleep(0.5)
            action = env.sample_action()
            obs, reward, done, truncated, _ = env.step(action)

            # print(obs, reward)

            eps_reward += np.mean(reward)
            n_steps += 1
            done = done | truncated

        status = "Done" if not truncated else "Truncated"
        print(f"Episode {i}\t Num Steps: {n_steps}\tReward: {eps_reward}\tStatus: {status}")

finally:
    env.close()