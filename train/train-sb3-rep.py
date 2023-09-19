# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:39:55 2023

@author: napat

Train the agent using Stable Baselines 3 Algorithms.
Repeat same configurations many times.
"""

import yaml, os
# import numpy as np
from envs.environment import DroneCatch
from stable_baselines3 import PPO, DQN, SAC, DDPG, TD3, A2C
from stable_baselines3.common.env_checker import check_env

# Identifier
run_name = "DQN01_7"

# Repetitive Experiments
num_runs = 30

# Import config file, Specify run name
config_file = f"config/{run_name.lower()}.yaml"
# run_name = "DQN01_5"
with open(config_file, "r") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

# Default directory to log
tensorboard_log=os.path.join("logs",run_name)
if config["model"]["tensorboard_log"] is None:
    config["model"]["tensorboard_log"] = tensorboard_log
 
## Creates environment
env = DroneCatch(**config['environment'])
# Check if environment follows OpenAI structure
check_env(env)

# Previous manual config:
# env = DroneCatch(resolution=(500, 500), frame_delay=5, reward_mult=2, dist_mult=0.1, obs_image=False)

## Create model
# Adjusts policy depending on whether observation is image-based
policy_type = "CnnPolicy" if config["environment"]["obs_image"] else "MlpPolicy"
# Model algorithm based on config
alg_dict = {"dqn":DQN, "ppo":PPO, "sac":SAC, "ddpg":DDPG, "td3":TD3, "a2c":A2C}    
policy = alg_dict[config["algorithm"]]

for i in range(num_runs):
# Creates the model
    model = policy(policy_type, env, **config["model"])
    
    # Start Training
    model.learn(**config["learn"])
    print("Finished learning Iteration: {:>2}".format(i+1))

# Saves Model
# run_name = config["model"]["tensorboard_log"].split("/")[-1]
m_file = os.path.join("model", run_name)
model.save(m_file)
print("Model saved to {}.zip".format(m_file))

