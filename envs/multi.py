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
                 distance_strategy: str = "minimum",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.distance_strategy = distance_strategy
        self.cardinal_prey = cardinal_prey
        self.num_preys = num_preys
        self.num_predators = num_predators

        self.prey = []
        self.predator = []
        self.action_space = []
        self.observation_space = []
        self.agent_list = []
        self.distances = []

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
            self.agent_list.append("prey")

        for i in range(self.num_predators):
            agent = Predator(canvas_size=self.canvas_shape,
                             icon_size=(self.icon_size, self.icon_size))
            self.predator.append(agent)
            self.action_space.append(agent.action_space)
            self.observation_space.append(obs_space)
            self.agent_list.append("predator")

        self.agents = [*self.prey, *self.predator]

    def step(self, action: List[int]):
        done, truncated = False, False
        info = {}

        self._move_agents(action)

        # Updates canvas
        self.draw_canvas()

        # Observation before termination
        obs = self.get_observation()

        # Check if termination conditions met
        if self.detect_collision():
            done = True
            info["is_success"] = True

        # Calculate reward
        reward = self.get_reward(done)

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

    def get_reward(self, done):
        """
        Get list of reward for each of the agent in the environment.
        """
        reward_list = []
        for i, agent_type in enumerate(self.agent_list):
            if agent_type == "prey":
                if done:
                    sign = -1.0
                else:
                    sign = 1.0
            else:
                if done:
                    sign = 1.0
                else:
                    sign = -1.0
            reward = sign * self.calculate_reward(done)
            reward_list.append(reward)
        return reward_list

    def _move_agents(self, actions: List[int]) -> None:
        for agent, action in zip(self.agents, actions):
            agent.move(action)

        # Move prey
        self.prey.move_in_circle(action_prey)

        # Move Predator
        delta = np.array([*self.convert_action(action_pred)]) * \
                self.move_speed
        self.predator.move(*delta)

    def calculate_reward(self, done) -> float:
        """
        Calculate current intermediate (non-terminal) reward for the agent.

        """
        if done:
            reward = self.reward_mult
        else:
            intermediate_reward = self.calculate_distance(normalise=True)
            reward = self.dist_mult * intermediate_reward

        return reward

    def precalculate_distance(self, normalise: bool = False):
        for prey in self.prey:
            for predator in self.predator:
                dist = super().calculate_distance(prey, predator, normalise=normalise)
                self.distances.append(dist)


    def calculate_distance(self, normalise: bool=False) -> float:
        if self.distance_strategy == "minimum":
            return min(self.distances)
        elif self.distance_strategy == "average":
            return np.mean(self.distances).item()
        elif self.distance_strategy == "sum":
            return sum(self.distances)
        else:
            pass

    def get_observation(self) -> np.ndarray:
        """
        Obtain observation at current time step.

        Returns
        -------
        obs : np.ndarray
            2D Array if obs_image, else (5,) 1D array.

        """
        if self.obs_image:
            obs = self.canvas
            return obs
        else:
            pred_pos = self.predator.get_position() / np.array(self.canvas_shape)
            prey_pos = self.prey.get_position() / np.array(self.canvas_shape)

            dist = np.linalg.norm(pred_pos - prey_pos)

            obs = np.r_[pred_pos, prey_pos, dist]

            return obs
