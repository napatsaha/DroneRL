# -*- coding: utf-8 -*-
"""
DroneCatch environment with multiple predators and/or preys.

All are active agents.
"""

from typing import List, Any, Optional, Union

import gymnasium as gym
from gymnasium import Env, Space, spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt

from envs.display import Predator, AngularPrey, CardinalPrey, Point
from envs.environment import DroneCatch


class MultiDrone(DroneCatch):
    """
    DroneCatch environment where both Predator and AngularPrey are able to learn simultaneously.
    
    Observations and Rewards returned will have an extra dimension:
        [predator, prey]
        
    Similarly, actions passed to step() must also have an extra dimension:
        [predator, prey]
    """
    agents: List[Point]
    observation_space = List[Space]
    action_space = List[Space]
    prey = List[Point]
    predator = List[Point]

    def __init__(self,
                 num_predators: int = 1,
                 num_preys: int = 1,
                 cardinal_prey: bool = True,
                 reward_distance_strategy: str = "individual-minimum",
                 observation_distance_strategy: str = "none",
                 distance_strategy: Optional = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_distance_strategy = observation_distance_strategy
        self.reward_distance_strategy = reward_distance_strategy
        # Compatibility
        if distance_strategy is not None:
            self.reward_distance_strategy = distance_strategy
        self.cardinal_prey = cardinal_prey
        self.num_preys = num_preys
        self.num_predators = num_predators

        self.prey = []
        self.predator = []
        self.action_space = []
        self.observation_space = []
        self.agent_list = []
        self.distances = {}

        for i in range(self.num_preys):
            if self.cardinal_prey:
                agent = CardinalPrey(
                    canvas_size=self.canvas_shape,
                    icon_size=(self.icon_size, self.icon_size),
                    speed=self.prey_move_speed)
            else:
                agent = AngularPrey(
                    self.canvas_shape,
                    angle_delta=self.prey_move_angle,
                    radius=round(self.radius * self.canvas_width / 2),
                    icon_size=(self.icon_size, self.icon_size)
                )
            obs_space = self._get_obs_space(agent_type=agent.name)
            self.prey.append(agent)
            self.agents.append(agent)
            self.action_space.append(agent.action_space)
            self.observation_space.append(obs_space)
            self.agent_list.append(f"{agent.name}{i+1}")

        for i in range(self.num_predators):
            agent = Predator(canvas_size=self.canvas_shape,
                             icon_size=(self.icon_size, self.icon_size),
                             speed=self.predator_move_speed)
            obs_space = self._get_obs_space(agent_type=agent.name)
            self.predator.append(agent)
            self.agents.append(agent)
            self.action_space.append(agent.action_space)
            self.observation_space.append(obs_space)
            self.agent_list.append(f"{agent.name}{i+1}")

        # self.agents = [*self.prey, *self.predator]

    def _get_obs_space(self, agent_type: str):
        num_obs = sum([
            # Coordinates
            2 * self.num_preys,
            2 * self.num_predators,
            self._get_obs_size(agent_type)
        ])
        obs_space = spaces.Box(
            low=np.zeros(num_obs),
            high=np.ones(num_obs),
            dtype=np.float64)
        return obs_space

    def _get_obs_size(self, agent_type: str) -> int:
        if self.observation_distance_strategy == "none":
            return 0
        elif self.observation_distance_strategy.endswith("all"):
            if self.observation_distance_strategy.startswith("global"):
                return self.num_predators * self.num_preys
            elif self.observation_distance_strategy.startswith("individual"):
                if agent_type == "predator":
                    return self.num_preys
                elif agent_type == "prey":
                    return  self.num_predators
        else:
            return 1

    def reset_position(self):
        self.precalculate_distance(normalise=True)

        # Loop to ensure no overlap between Predator and AngularPrey
        while True:
            for object in self.agents:
                if object.name == "prey" and self.random_prey:
                    object.randomise_position()
                elif object.name == "predator" and self.random_predator:
                    object.randomise_position()
                else:
                    object.reset_position()

            distance = self.calculate_distance(agent_name=None, strategy="global-minimum")

            if not self.detect_collision() and distance > self.min_distance:
                break

    def step(self, action: List[int]):
        done, truncated = False, False
        info = {}

        self._move_agents(action)

        # Updates canvas
        self.draw_canvas()

        self.precalculate_distance(normalise=True)

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

    def _move_agents(self, actions: List[int]) -> None:
        for agent, action in zip(self.agents, actions):
            agent.move(action)

    def get_reward(self, done: bool = False):
        """
        Get list of reward for each of the agent in the environment.
        """
        # self.precalculate_distance(normalise=True)

        reward_list = []
        for agent, agent_name in zip(self.agents, self.agent_list):
            if agent.name == "prey":
                if done:
                    sign = -1.0
                else:
                    sign = 1.0
            else:
                if done:
                    sign = 1.0
                else:
                    sign = -1.0
            reward = sign * self.calculate_each_reward(agent_name, done)
            reward_list.append(reward)
        return reward_list

    def calculate_each_reward(self, agent_name, done: bool = False) -> float:
        """
        Calculate individual reward for a specific agent
        """
        if done:
            reward = self.reward_mult
        else:
            intermediate_reward = self.calculate_distance(agent_name)
            reward = self.dist_mult * intermediate_reward

        return reward

    def precalculate_distance(self, normalise: bool = True):
        # self.distances = []
        # for prey in self.prey:
        #     for predator in self.predator:
        #         dist = super().calculate_distance(prey, predator, normalise=normalise)
        #         self.distances.append(dist)
        self.distances = {}
        for i, prey in enumerate(self.prey):
            for j, predator in enumerate(self.predator):
                dist = super().calculate_distance(prey, predator, normalise=normalise)
                src = f"prey{i+1}"
                des = f"predator{j+1}"
                self.distances[(src, des)] = dist

    def calculate_distance(self,
                           agent_name: Union[str, None],
                           strategy: str = None) -> Union[float, list]:
        """
        Calculate distance for a specific agent, according to one of the following strategies:
        - minimum: Take the shortest distance between each of the predators and each of the preys
        - average: Take the mean predator-prey distances.
        - sum: Take the total of all predator-prey distances.
        - individual: (Not implemented yet; perhaps some form of distance that is only related to
            the agent in question)
        """
        # self.precalculate_distance(normalise)

        if strategy is None:
            strategy = self.reward_distance_strategy

        # if strategy == "global-minimum":
        #     return min(self.distances)
        # elif strategy == "global-average":
        #     return np.mean(self.distances).item()
        # elif strategy == "global-sum":
        #     return sum(self.distances)

        # Collating relevant distances
        if strategy.startswith("global"):
            distances = list(self.distances.values())
        elif strategy.startswith("individual"):
            distances = []
            for object_names, dist in self.distances.items():
                if agent_name in object_names:
                    distances.append(dist)
        else:
            return 1.0

        # Aggregation
        if strategy.endswith("minimum") or strategy.endswith("closest"):
            return min(distances)
        elif strategy.endswith("average"):
            return float(np.mean(distances))
        elif strategy.endswith("sum"):
            return sum(distances)
        elif strategy.endswith("all"):
            return distances
        else:
            raise Exception("Invalid distance strategy")



    # def calculate_distance(self,
    #                        object1: Point=None, object2: Point=None,
    #                        normalise: bool=False) -> float:
    #     """
    #     Calculate the minimum distance out of all predator-prey pairs.
    #
    #     Used in reset_position().
    #
    #     Parameters
    #     ----------
    #     object1
    #     object2
    #     normalise
    #
    #     Returns
    #     -------
    #     float
    #     """
    #     return self.calculate_distance(agent=None, strategy="minimum",
    #                                         normalise=normalise)

    def get_each_observation(self, agent):
        obs = []

        # Position-based observations
        for object in self.agents:
            pos = object.get_position(normalise=True)
            obs.extend(pos)

        # Distance-based observations
        if self.observation_distance_strategy != "none":
            dist_obs = self.calculate_distance(
                agent,
                strategy=self.observation_distance_strategy
            )
            if isinstance(dist_obs, list):
                obs.extend(dist_obs)
            elif isinstance(dist_obs, float):
                obs.append(dist_obs)

        return np.array(obs)

    def get_observation(self) -> np.ndarray:
        """
        """
        obs = []
        for agent in self.agent_list:
            obs.append(self.get_each_observation(agent))
        return obs

    def detect_collision(self) -> bool:
        for prey in self.prey:
            for predator in self.predator:
                if super().detect_collision(prey, predator):
                    return True
        return False

    def sample_action(self):
        actions = []
        for space in self.action_space:
            actions.append(space.sample())
        return actions
