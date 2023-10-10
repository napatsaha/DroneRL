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

from envs.display import Predator, AngularPrey
from envs.environment import DroneCatch
from envs.display import CardinalPrey


class DualDrone(DroneCatch):
    """
    DroneCatch environment where both Predator and AngularPrey are able to learn simultaneously.
    
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

        self._move_agents(action_prey, action_pred)

        # # Move prey
        # self.prey.move_in_circle(action_prey)
        #
        # # Move Predator
        # delta = np.array([*self.convert_action(action_pred)]) * \
        #         self.move_speed
        # self.predator.move_to_position(*delta)

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

    def _move_agents(self, action_prey, action_pred):
        # Move prey
        self.prey.move_in_circle(action_prey)

        # Move Predator
        delta = np.array([*self.convert_action(action_pred)]) * \
                self.move_speed
        self.predator.move_to_position(*delta)
        
        
class DualLateral(DualDrone):
    """
    Allows prey to move_to_position into cardinal directions instead of circularly.
    """

    def __init__(self, prey_move_speed = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        delattr(self, "prey_move_angle")
        # del self.prey_move_angle

        self.prey_move_speed = prey_move_speed
        self.action_space = [spaces.Discrete(5) for _ in range(2)]

        self._predator_speed = round(self.predator_move_speed * 0.01 * self.canvas_width)
        self._prey_speed = round(self.prey_move_speed * 0.01 * self.canvas_width)

        self.prey = CardinalPrey(self.canvas_shape,
                                 icon_size=(self.icon_size, self.icon_size))

        self.agents = [self.prey, self.predator]

    def _move_agents(self, action_prey, action_pred):
        # Move prey
        delta = np.array([*self.convert_action(action_prey)]) * \
                self._prey_speed
        self.prey.move_to_position(*delta)

        # Move Predator
        delta = np.array([*self.convert_action(action_pred)]) * \
                self._predator_speed
        self.predator.move_to_position(*delta)

    def randomise_prey_position(self):
        rand_pos = np.random.randint(self.canvas_shape)
        self.prey.reset_position(*rand_pos)