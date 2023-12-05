# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:52:20 2023

@author: napat
"""

from typing import List, Union, Optional

import gymnasium as gym
from gymnasium import Env, Space, spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt

from envs.display import Predator, AngularPrey, Mover, CardinalPrey
from envs.geometry import Canvas, LineSegment, Point, convert_line_to_box, read_obstacle_config
from utils.tools import safe_simplify


class DroneCatch(Env):
    """
    Predator-AngularPrey Drone gym environment for reinforcement learning.
    
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    version = "v1"

    agents: List[Mover]
    active_agents: List[Mover]
    nonactive_agents: List[Mover]
    observation_space = Union[Space, List[Space]]
    action_space = Union[Space, List[Space]]
    prey = Union[Mover, List[Mover]]
    predator = Union[Mover, List[Mover]]

    def __init__(self,
                 num_preys: int = 0,
                 num_predators: int = 1,
                 random_action: bool=True,
                 obs_image: bool=False,
                 resolution: int=800,
                 icon_scale: float=0.1,
                 cardinal_prey: bool = True,
                 predator_move_speed: float=5,
                 prey_move_speed: float=5,
                 prey_move_angle: float=5,
                 radius: float=0.8,
                 random_prey: bool=True,
                 random_predator: bool=True,
                 predator_spawn_area: tuple = None,
                 prey_spawn_area: tuple = None,
                 min_distance: float = 0,
                 verbose: int = 0,
                 dist_mult: float=0.1,
                 intermediate_reward: bool=True,
                 reward_mult: float=1.0,
                 trunc_limit: int=100,
                 show_rays: bool=False,
                 num_rays: int=8,
                 num_buffers: int = 2,
                 obstacle_file: str=None,
                 frame_delay: int=50,
                 render_mode: str="human",
                 manual_control: bool=False,
                 diagnostic: bool=False):
        """
        Create the environment.

        Parameters
        ----------
        obs_image : bool, optional
            Whether or not to use screen image as observation. If False, only positional information are available to the agent. The default is False.
        resolution : int, optional
            Screen resolution (width and height). The default is 800.
        icon_scale : float, optional
            Size of both predator and prey icons in relation to screen width/height. The default is 0.1.
        prey_move_angle : int, optional
            Amount of angles (in degrees) that the prey will move_to_position in each timestep (counter-clockwise). The default is 5.
        predator_move_speed : int, optional
            The speed at which the predator moves. The default is 5.
        radius : float, optional
            Radius of the circle of prey movement. The default is 0.8.
        dist_mult : float, optional
            Weight given to consecutive negative reward in each timestep to the agent based inversely on distance between predator and prey. The default is 0.1.
        reward_mult : float, optional
            Weight given to final positive reward when the predator catches the prey. The default is 1.0.
        trunc_limit : int, optional
            maximum timestep per episode before truncation. The default is 100.
        frame_delay : int, optional
            Time (in ms) to wait between frames when render_mode = human. The default is 50.
        render_mode : str, optional
            Type of rendering mode: supports human or rgb_array. The default is "human".

        Returns
        -------
        None.

        """
        super(DroneCatch, self).__init__()

        # Fields
        self.obs_image = obs_image
        self.num_predators = num_predators
        self.num_preys = num_preys

        # Raycasting
        self.num_rays = num_rays
        self.rays = []
        self.show_rays = show_rays
        self.diagnostic = diagnostic
        self.verbose = verbose

        # Build a canvas
        self.resolution = resolution if isinstance(resolution, int) else resolution[0]
        self.canvas_shape = (resolution, resolution) if isinstance(resolution, int) else resolution
        self.canvas_width = self.resolution
        self.max_width = np.sqrt(self.canvas_shape[0]**2 + self.canvas_shape[1]**2) # For normalising
        self.canvas = Canvas(*self.canvas_shape)

        # Icon and Move speed
        self.icon_scale = icon_scale
        self.icon_size = round(icon_scale * self.canvas_width)
        self.move_speed = round(predator_move_speed * 0.01 * self.canvas_width)
        self.num_buffers = num_buffers
        self.random_action = random_action

        # Parameters related to spawning
        self.random_prey = random_prey
        self.random_predator = random_predator
        self.prey_spawn_area = prey_spawn_area
        self.predator_spawn_area = predator_spawn_area
        self._verify_min_distance(min_distance, resolution)

        ##############
        # Prey, Predator and observation, action spaces
        self.prey = []
        self.predator = []
        self.action_space = []
        self.observation_space = []
        self.agents = []
        self.active_agents = []
        self.nonactive_agents = []
        self.agent_list = []

        # Predator/AngularPrey configurations
        self.predator_move_speed = predator_move_speed
        self.prey_move_speed = prey_move_speed
        self.cardinal_prey = cardinal_prey
        # For non cardinal movement
        self.prey_move_angle = prey_move_angle
        self.radius = radius

        if self.obs_image:
            obs_space = spaces.Box(
                low = np.zeros(self.canvas_shape),
                high= np.ones(self.canvas_shape),
                dtype=np.float64
            )
        else:
            obs_space = spaces.Box(
                low=np.zeros(2 * self.num_rays),
                high=np.ones(2 * self.num_rays),
                dtype=np.float64)

        # Initialises Predator and AngularPrey classes
        for i in range(max(1, self.num_preys)):
            if self.cardinal_prey:
                agent = CardinalPrey(
                    canvas_size=self.canvas_shape,
                    icon_size=(self.icon_size, self.icon_size),
                    speed=self.prey_move_speed,
                    spawn_area=self.prey_spawn_area,
                    num_buffers=self.num_buffers
                )
            else:
                agent = AngularPrey(
                    self.canvas_shape,
                    angle_delta=self.prey_move_angle,
                    radius=round(self.radius * self.canvas_width / 2),
                    icon_size=(self.icon_size, self.icon_size),
                    num_buffers=self.num_buffers
                )

            self.prey.append(agent)
            self.agents.append(agent)
            if self.num_preys > 0:
                agent.set_active()
                agent.set_diagnostic(self.diagnostic)
                self.active_agents.append(agent)
                self.action_space.append(agent.action_space)
                self.observation_space.append(obs_space)
                self.agent_list.append(f"{agent.name}{i + 1}")
            else:
                self.nonactive_agents.append(agent)

        for i in range(max(1, self.num_predators)):
            agent = Predator(
                canvas_size=self.canvas_shape,
                icon_size=(self.icon_size, self.icon_size),
                speed=self.predator_move_speed,
                spawn_area=self.predator_spawn_area,
                num_buffers=self.num_buffers
            )
            self.predator.append(agent)
            self.agents.append(agent)
            if self.num_predators > 0:
                agent.set_active()
                agent.set_diagnostic(self.diagnostic)
                self.active_agents.append(agent)
                self.action_space.append(agent.action_space)
                self.observation_space.append(obs_space)
                self.agent_list.append(f"{agent.name}{i + 1}")
            else:
                self.nonactive_agents.append(agent)

        self.observation_space = safe_simplify(self.observation_space)
        self.action_space = safe_simplify(self.action_space)

        ###########
        # Obstacles
        self.obstacle_list = []
        self._populate_obstacles(obstacle_file)
        self._update_obstacles() # Updates on each agent

        # Episode Control variables
        self.trunc_limit = trunc_limit
        self.timesteps = 0

        # Learning/Rewards Variables
        self.dist_mult = dist_mult
        self.reward_mult = reward_mult
        self.intermediate_reward = intermediate_reward

        # Render Mode
        self.frame_delay = frame_delay
        self.render_mode = render_mode
        self.manual_control = manual_control
        self.key_dict = {
            32     : 0, # Space Arrow
            2490368: 1, # Up Arrow
            2424832: 2, # Left Arrow
            2621440: 3, # Down Arrow
            2555904: 4, # Right Arrow
        }

    def _verify_min_distance(self, min_distance, resolution):
        # To prevent infinite while loops
        if self.predator_spawn_area is not None or self.prey_spawn_area is not None:
            self.min_distance = 0.0
            return
        if min_distance <= 1:
            assert min_distance >= 0, "Minimum spawn distance should be between 0 and 1, or as pixels less than resolution."
            self.min_distance = min_distance
        else:
            assert min_distance < resolution, "Minimum spawn distance should be between 0 and 1, or as pixels less than resolution."
            self.min_distance = float(min_distance) / resolution

    def _update_obstacles(self):
        for agent in self.agents:
            agent.clear_obstacle()
            # Add obstacles
            agent.add_obstacle(self.obstacle_list)
            # Add other agents
            agent.add_obstacle([ag for ag in self.agents if ag != agent])

    def _populate_obstacles(self, file: Optional[str]):
        # line = LineSegment(Point(0.3 * self.canvas_width, 0.6 * self.canvas_width),
        #                    Point(0.7 * self.canvas_width, 0.2 * self.canvas_width))
        # line2 = LineSegment(Point(0.5 * self.canvas_width, 0.5 * self.canvas_width),
        #                     Point(0.8 * self.canvas_width, 0.8 * self.canvas_width))
        # for obs in (line, line2):
        #     boxed_obs = convert_line_to_box(obs, self.canvas_width * 0.01)
        #     self.obstacle_list.extend(boxed_obs)
        if file is not None:
            obstacles = read_obstacle_config(file, self.canvas)
            self.obstacle_list.extend(obstacles)
        # pass

    def reset(self):

        # Reset Positions for Predator and AngularPrey
        # for elem in self.agents:
        #     elem.reset_position()

        # Randomise if necessary
        self.reset_position()

        # Get observations and Rays from radial raycasting if necessary
        obs = self.get_observation()

        # Imprints new positions onto environment canvas
        self.draw_canvas()

        self.timesteps = 0

        return obs, {}

    def reset_position(self):

        # Loop to ensure no overlap between Predator and AngularPrey
        while True:
            for object in self.agents:
                if object.name == "prey" and self.random_prey:
                    object.randomise_position()
                elif object.name == "predator" and self.random_predator:
                    object.randomise_position()
                else:
                    object.reset_position()

            distance = self.calculate_distance(normalise=True)

            if not self.detect_collision() and distance > self.min_distance:
                # if self.verbose == 3:
                #     print(f"{distance:.3f}")
                # elif self.verbose == 4:
                #     distance = self.calculate_distance(normalise=False)
                #     print(f"{distance:.1f}")
                break

    def step(self, action: Union[List[int], int]) -> \
            tuple[
                Union[np.ndarray, List[np.ndarray]],
                Union[float, List[float]],
                bool,
                bool,
                dict
            ]:
        """
        Take an action and transition the environment to the next step.
        
        Predator moves by the provided action; AngularPrey moves in a constant trajectory
        around a circle counter-clockwise.

        Parameters
        ----------
        action : int
            Direction to move_to_position the Predator (range [0,4]).

        Returns
        -------
        obs : np.ndarray
            Observation.
        reward : float
            reward.
        done : bool
            Whether the environment terminates normally.
        truncated : bool
            Whether the maximum number of steps is reached.
        info : dict
            Contains information about whether predator was successful in this 
            episode ("is_success"). Only available on terminal step.

        """
        reward = 0.0
        done, truncated = False, False
        info = {}

        self._move_agents(action)

        # Calculate reward
        reward = self.get_reward(terminal=False)

        # Observation before termination
        obs = self.get_observation()

        # Updates canvas
        self.draw_canvas()

        ## Reset episode if termination conditions met
        # Check for collision
        if self.detect_collision():
            # self.reset()
            reward = self.get_reward(terminal=True)
            done = True
            info["is_success"] = True

        # Check if Number of Steps exceed Truncation Limit
        self.timesteps += 1
        if self.timesteps >= self.trunc_limit:
            # self.reset()
            truncated = True
            info["is_success"] = False

        return obs, reward, done, truncated, info

    def get_reward(self, terminal: bool=False):

        if terminal:
            # reward = 1.0 * self.reward_mult
            terminal_reward = [1.0 if ag.name == "predator" else -1.0 for ag in self.active_agents]
            reward = self.reward_mult * np.array(terminal_reward)
        else:
            # reward = self.dist_mult * self.calculate_reward()
            reward = self._intermediate_reward()

            reward = self.dist_mult * np.array(reward)

        return safe_simplify(reward)

    def _intermediate_reward(self):
        reward = []
        for ag in self.active_agents:
            if self.intermediate_reward:
                intermediate = self.calculate_distance(normalise=True)
            else:
                intermediate = 1.0

            if ag.name == "predator":
                reward.append(-intermediate)
            else:
                reward.append(intermediate)
        return reward

    def _move_agents(self, actions: Union[int, List[int]]):
        if not isinstance(actions, (list, tuple, np.ndarray)):
            actions = [actions]

        assert len(self.active_agents) == len(actions), \
            "Number of received actions and active agents in" \
            "the environment do not match."

        for agent, action in zip(self.active_agents, actions):
            agent.move(action)

        for agent in self.nonactive_agents:
            action = agent.sample_action() if self.random_action else 0.0
            agent.move(action)

    def render(self) -> Union[int, np.ndarray, None]:

        if self.render_mode == "human":
            key = self.canvas.show(frame_delay=self.frame_delay, manual=self.manual_control)
            if self.manual_control:
                action = self.key_dict[key]
                return action
            else:
                return None
        elif self.render_mode == "rgb_array":
            return self.canvas.canvas

    def close(self):
        self.canvas.close()

    def draw_canvas(self):
        """
        Draw a frame of canvas based on active elements.
        """
        # Initialise canvas
        self.canvas.clear()

        # Draw obstacles
        for obstacle in self.obstacle_list:
            self.canvas.draw(obstacle)

        # Draw icons
        for icon in self.agents:
            self.canvas.draw(icon)

        if self.show_rays:
            self.canvas.draw(self.rays)

    def sample_action(self) -> List[int]:
        """
        Sample a random action from each of the active agents.
        """
        actions = []
        for agent in self.active_agents:
            a = agent.sample_action()
            actions.append(a)
        return actions

    def detect_collision(self) -> bool:
        for prey in self.prey:
            for predator in self.predator:
                has_collided = self._detect_collision_between_objects(prey, predator)
                if has_collided:
                    return True
        return False


    def _detect_collision_between_objects(self, object1: Mover, object2: Mover) -> bool:
        """
        Detect whether two drones are in contact with each other (overlapping).

        Returns
        -------
        bool
            Collided or not.

        """
        x_collided = np.abs(object1.x - object2.x) <= (object1.icon_w/2 + object2.icon_w/2)
        y_collided = np.abs(object1.y - object2.y) <= (object1.icon_h/2 + object2.icon_h/2)

        return x_collided & y_collided

    def calculate_distance(self, normalise: bool = True) -> float:
        """
        Calculate globally-shortest distance between predators and preys.
        """
        distances = []
        for i, prey in enumerate(self.prey):
            for j, predator in enumerate(self.predator):
                dist = self._distance_between_objects(prey, predator, normalise=normalise)
                distances.append(dist)

        return min(distances)

    def _distance_between_objects(self,
                                  object1: Mover, object2: Mover,
                                  normalise: bool = False) -> float:
        """
        Calculate Euclidean distance between two Mover objects.
        """
        scale = np.array(self.canvas_shape) if normalise else 1.0
        pos1 = object1.get_position() / scale
        pos2 = object2.get_position() / scale

        dist = np.linalg.norm(pos1 - pos2)

        return dist

    def calculate_reward(self) -> float:
        """
        Calculate current intermediate (non-terminal) reward for the agent.

        Returns
        -------
        float
            Intermediate reward.

        """
        intermediate_reward = - self.calculate_distance() / self.canvas_width

        return intermediate_reward

    def get_observation(self) -> List[np.ndarray]:
        """
        Obtain observation at current time step.

        Returns
        -------
        obs : np.ndarray
            2D Array if obs_image, else (5,) 1D array.

        """
        if self.show_rays:
            self.rays = []
        observation = []
        for agent in self.active_agents:
            if self.obs_image:
                obs = self.canvas.canvas
                observation.append(obs)
            else:
                # Radial raycasting to obtain distances and points of contact
                obs, rays, obj_types = agent.radial_raycast(
                    agent.obstacle_list, self.canvas,
                    return_rays=self.show_rays,
                    num_rays=self.num_rays
                )

                # print(rays)

                collision_dict = {
                    "obstacle": 0,
                    "prey": 1,
                    "predator": -1
                }

                # Convert and Normalise values
                obs = obs / self.max_width
                obj_types = np.vectorize(collision_dict.get)(obj_types)
                obs = np.r_[obs, obj_types]
                observation.append(obs)

                if self.show_rays:
                    self.rays.extend(rays)

        return safe_simplify(observation)

    def set_frame_delay(self, frame_delay):
        self.frame_delay = frame_delay


def spawn_in_area(x1, y1, x2, y2):
    pass