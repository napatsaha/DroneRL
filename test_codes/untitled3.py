# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:37:23 2023

@author: napat
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import torch as th

file = "logs/test1/DQN_1/progress.csv"

progress = pd.read_csv(file)
progress.info()

plt.plot(progress['time/total_timesteps'], progress['rollout/ep_len_mean'])

env.frame_delay = 15
rew_list = []
for i in range(50):
    ep_rew = 0
    done = False
    obs, _ = env.reset()
    while not done:
        env.render()
        action = policy.predict(obs)
        obs, rew, done, trunc, info = env.step(action)
        done = done | trunc
        ep_rew += rew
    rew_list.append(ep_rew)
    print(f"Episode {i+1}: Reward: {ep_rew:.4f}")
env.close()


