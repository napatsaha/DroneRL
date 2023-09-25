# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:18:24 2023

@author: napat

Evaluates agent using previously trained model
"""

import yaml, os
import numpy as np
from envs.environment import DroneCatch
from stable_baselines3 import PPO, DQN, SAC, DDPG, TD3, A2C
from stable_baselines3.common.env_checker import check_env

# Specify run_name to load from
run_name = "DQN01_5"

# Import config file, Specify run name
config_file = "config/{}.yaml".format(run_name.lower())

with open(config_file, "r") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

# # Default directory to log
# tensorboard_log="./logs"
# if "tensorboard_log" in dir():
#     config["model"]["tensorboard_log"] = tensorboard_log

## Creates environment
# Previous manual config:
# env = DroneCatch(resolution=(500, 500), frame_delay=5, reward_mult=2, dist_mult=0.1, obs_image=False)
env = DroneCatch(**config['environment'])

# Import Model
model = DQN.load(os.path.join("model",run_name), env=env)

# Simulation Test
np.random.seed(1234)
num_eps = 50
vec_env = model.get_env()
vec_env.envs[0].env.frame_delay = 20
try:
    successes = np.empty((num_eps, ), dtype=np.bool_)
    for i in range(num_eps):
        done = False
        obs = vec_env.reset()
        eps_reward, n_steps = 0,0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render("human")
            # print(obs, reward)
            
            eps_reward += reward
            n_steps += 1
            # done = done | truncated
        
        successes[i] = info[0]["is_success"]
    
        # status = "Done" if not truncated else "Truncated"
        print(f"Episode {i}\t Num Steps: {n_steps}\tReward: {eps_reward}")

    print(f"\nSuccess Rate:\t{successes.mean():.0%}")
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")

finally:
    vec_env.close()