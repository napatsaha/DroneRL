# -*- coding: utf-8 -*-
"""
DroneCatch environment with multiple predators and/or preys.

All are active agents.
"""

from typing import List

import gymnasium as gym
from gymnasium import Env, Space, spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt

from envs.display import Predator, AngularPrey, CardinalPrey
from envs.environment import DroneCatch


class MultiDrone(DroneCatch):
    """
    DroneCatch environment where both Predator and AngularPrey are able to learn simultaneously.
    
    Observations and Rewards returned will have an extra dimension:
        [predator, prey]
        
    Similarly, actions passed to step() must also have an extra dimension:
        [predator, prey]
    """
    observation_space = List[Space]
    action_space = List[Space]
    prey = List
    predator = List

    def __init__(self,
                 num_predators: int = 1,
                 num_preys: int = 1,
                 cardinal_prey: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cardinal_prey = cardinal_prey
        self.num_preys = num_preys
        self.num_predators = num_predators

        self.prey = []
        self.predator = []
        self.action_space = []
        self.observation_space = []

        num_obs = sum([
            # Coordinates
            2 * self.num_preys,
            2 * self.num_predators
        ])

        obs_space = spaces.Box(
                low = np.zeros(num_obs),
                high = np.ones(num_obs),
                dtype = np.float64)

        for i in range(self.num_preys):
            if self.cardinal_prey:
                agent = CardinalPrey(
                    canvas_size=self.canvas_shape,
                    icon_size=(self.icon_size, self.icon_size))
            else:
                agent = AngularPrey(
                    self.canvas_shape,
                    angle_delta=self.prey_move_angle,
                    radius=round(self.radius * self.canvas_width / 2),
                    icon_size=(self.icon_size, self.icon_size)
                )
            self.prey.append(agent)
            self.action_space.append(agent.action_space)
            self.observation_space.append(obs_space)

        for i in range(self.num_predators):
            agent = Predator(canvas_size=self.canvas_shape,
                             icon_size=(self.icon_size, self.icon_size))
            self.predator.append(agent)
            self.action_space.append(agent.action_space)
            self.observation_space.append(obs_space)

        self.agents = [*self.prey, *self.predator]

    def step(self, action: List[int]):
        done, truncated = False, False
        info = {}

        self._move_agents(action)

        # Updates canvas
        self.draw_canvas()

        # Calculate reward
        reward = self.calculate_reward()
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

    def _move_agents(self, action):
        # Move prey
        self.prey.move_in_circle(action_prey)

        # Move Predator
        delta = np.array([*self.convert_action(action_pred)]) * \
                self.move_speed
        self.predator.move(*delta)
        
        
