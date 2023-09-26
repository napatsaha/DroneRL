# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:52:20 2023

@author: napat
"""

from typing import List
import gymnasium as gym
from gymnasium import Env, Space, spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt

from envs.display import Predator, Prey
from envs.environment import DroneCatch


class DualDrone(DroneCatch):
    """
    DroneCatch environment where both Predator and Prey are able to learn simultaneously.
    
    Observations and Rewards returned will have an extra dimension:
        [predator, prey]
        
    Similarly, actions passed to step() must also have an extra dimension:
        [predator, prey]
    """
    observation_space = List[Space]
    action_space = List[Space]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.action_space = [spaces.Discrete(5), spaces.Discrete(3)]
        
        self.observation_space = [self.observation_space for _ in range(2)]
    
    def step(self, action: List[int]):
        done, truncated = False, False
        info = {}

        action_prey = action[1]
        action_pred = action[0]

        # Move prey
        self.prey.move_in_circle(action_prey)

        # Move Predator
        delta = np.array([*self.convert_action(action_pred)]) * \
                self.move_speed
        self.predator.move(*delta)

        # Updates canvas
        self.draw_canvas()

        # Calculate reward
        reward_pred = self.dist_mult * self.calculate_reward()
        reward_prey = self.dist_mult * (- self.calculate_reward())

        # Observation before termination
        obs = self.get_observation()

        ## Reset episode if termination conditions met
        # Check for collision
        if self.detect_collision():
            # self.reset()
            reward_pred = 1.0 * self.reward_mult
            reward_prey = -1.0 * self.reward_mult
            done = True
            info["is_success"] = True

        reward = [reward_pred, reward_prey]

        # Check if Number of Steps exceed Truncation Limit
        self.trunc_count += 1
        if self.trunc_count >= self.trunc_limit:
            # self.reset()
            truncated = True
            info["is_success"] = False

        return obs, reward, done, truncated, info

    def get_observation(self) -> List[np.ndarray]:
        obs = super().get_observation()
        return [obs, obs]
        
        
    
    
    