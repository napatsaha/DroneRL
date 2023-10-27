# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 20:05:16 2023

@author: napat
"""

import os
import torch as th
import numpy as np
import gymnasium as gym
from envs.environment_v0 import DroneCatch
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import CSVOutputFormat, configure
from stable_baselines3.common.utils import configure_logger

from algorithm.network import QNetwork
from algorithm.policy import DQNPolicy
from envs.mountain_car import MountainCarWrapper


# env = DroneCatch(reward_mult=10)
env = gym.make("MountainCar-v0")
env = MountainCarWrapper(env)
# state, _ = env.reset()
run_name = "MountainCar"
run_dir = "test2"

# agent1 = DQN("MlpPolicy", env=env)
# agent2 = DQN("MlpPolicy", env=env)

# agent1.train(gradient_steps=1)
# agent1.learn(0)


# buffer = ReplayBuffer(100, env.observation_space, env.action_space)
# buffer

# q_net = QNetwork(env.observation_space, env.action_space)
# q_net(state)

policy = DQNPolicy(
    env.observation_space, env.action_space,
    train_freq=16,
    gradient_steps=8,
    batch_size=128,
    buffer_size=100000,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    target_update_interval=600,
    learning_rate=4e-3,
    net_kwargs=dict(net_arch=[256,256]),
    total_timesteps=int(2e5),
    log_interval=10,
    log_output=["csv", "stdout"],
    log_name=run_name,
    log_dir=f"logs/{run_dir}"
)



learning_starts = 10000
e = 0
while not policy.done:
    n_steps = 0
    eps_reward = []
    state, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action = policy.predict(state)
        nextstate, reward, done, truncated, info = env.step(action)
        policy.store_transition(state, nextstate, action, reward, done, truncated, info)
        
        if policy.num_timesteps > learning_starts and policy.num_timesteps % policy.train_freq == 0:
            policy.train()
            
        policy.step()
        
        state = nextstate
        n_steps += 1
        eps_reward.append(reward)
    # print(f"Episode {e+1}:\tMean Reward {np.mean(eps_reward):.3f}\tFinal Reward: {reward:.3f}\tNum Steps: {n_steps}")
    e+=1
env.close()

if not os.path.exists(os.path.join("model",run_dir)):
    os.mkdir(f"model/{run_dir}")

th.save(policy.q_net.state_dict(), f"model/{run_dir}/{policy.run_name}.pt")
