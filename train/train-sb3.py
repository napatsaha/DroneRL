# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:39:55 2023

@author: napat

Train the agent using Stable Baselines 3 Algorithms
"""

import yaml, os
# import numpy as np
from envs.environment import DroneCatch
from stable_baselines3 import PPO, DQN, SAC, DDPG, TD3, A2C
from stable_baselines3.common.env_checker import check_env

# Import config file, Specify run name
config_file = "config/dqn01_5.yaml"
run_name = "DQN01_5"
with open(config_file, "r") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

# Default directory to log
tensorboard_log="./logs"
if "tensorboard_log" in dir():
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
# Creates the model
model = policy(policy_type, env, **config["model"])

# Start Training
model.learn(tb_log_name=run_name, **config["learn"])
print("Finished learning")

# Saves Model
m_file = os.path.join("model", run_name)
model.save(m_file)
print("Model saved to {}.zip".format(m_file))

# # Simulation Test
# np.random.seed(1234)
# num_eps = 50
# vec_env = model.get_env()
# vec_env.envs[0].env.frame_delay = 20
# try:
    
#     for i in range(num_eps):
#         done = False
#         obs = vec_env.reset()
#         eps_reward, n_steps = 0,0
#         while not done:
#             action, _state = model.predict(obs, deterministic=True)
#             obs, reward, done, info = vec_env.step(action)
#             vec_env.render("human")
#             # print(obs, reward)
            
#             eps_reward += reward
#             n_steps += 1
#             # done = done | truncated
        
#         # status = "Done" if not truncated else "Truncated"
#         print(f"Episode {i}\t Num Steps: {n_steps}\tReward: {eps_reward}")

# # for i in range(1000):
# #     action, _state = model.predict(obs, deterministic=True)
# #     obs, reward, done, info = vec_env.step(action)
# #     vec_env.render("human")

# finally:
#     vec_env.close()