# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:22:37 2023

@author: napat
"""
import os
import torch as th
import gymnasium as gym
from algorithm.policy import DQNPolicy
from stable_baselines3.common.type_aliases import GymEnv

class DQNAgent:
    """
    Simple learner where the environment is tied to one agent.
    """
    def __init__(
            self,
            env: GymEnv,

    ):
        self.policy = DQNPolicy(env.observation_space, env.action_space,
            buffer_size=100000,
            total_timesteps=100000, log_output=["csv", "stdout"],
            exploration_fraction=0.3, log_name="MountainCar",
            log_dir="logs/test1")
        self.env = env

    def learn