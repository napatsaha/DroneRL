# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:52:20 2023

@author: napat
"""

from typing import List, Union

import gymnasium as gym
from gymnasium import Env, Space, spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt

from envs.display import Predator, AngularPrey, Mover, CardinalPrey
from envs.geometry import Canvas, LineSegment, Point, convert_line_to_box


class DroneCatch(Env):
    """
    Predator-AngularPrey Drone gym environment for reinforcement learning.
    
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    agents = List[Mover]
    
    def __init__(self,
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
                 min_distance: float = 0,
                 verbose: int = 0,
                 dist_mult: float=0.1,
                 reward_mult: float=1.0,
                 trunc_limit: int=100,
                 show_rays: bool=False,
                 num_rays: int=8,
                 frame_delay: int=50,
                 render_mode: str="human",
                 manual_control: bool=False):
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
        self.num_rays = num_rays
        self.rays = []
        self.show_rays = show_rays
        self.cardinal_prey = cardinal_prey
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
        
        # Define action space (4 directions + stationary)
        self.action_space = spaces.Discrete(5,)
        
        # Define observation space (xy for prey and predator and distance)
        self.obs_image = obs_image
        if self.obs_image:
            self.observation_space = spaces.Box(
                np.zeros(self.canvas_shape), 
                np.ones(self.canvas_shape),
                dtype=np.float64)
        else:
            self.observation_space = spaces.Box(
                low = np.zeros(5),
                high = np.ones(5),
                dtype = np.float64)
        
        # Predator/AngularPrey configurations
        self.predator_move_speed = predator_move_speed
        self.prey_move_speed = prey_move_speed
        # For non cardinal movement
        self.prey_move_angle = prey_move_angle
        self.radius = radius

        # Parameters related to spawning
        self.random_prey = random_prey
        self.random_predator = random_predator
        if min_distance <= 1:
            assert min_distance >= 0, "Minimum spawn distance should be between 0 and 1, or as pixels less than resolution."
            self.min_distance = min_distance
        else:
            assert min_distance < resolution, "Minimum spawn distance should be between 0 and 1, or as pixels less than resolution."
            self.min_distance = float(min_distance) / resolution

        # Obstacles
        self.obstacle_list = []
        self._populate_obstacles()

        # Initialises Predator and AngularPrey classes
        if self.cardinal_prey:
            self.prey = CardinalPrey(
                canvas_size=self.canvas_shape,
                icon_size=(self.icon_size, self.icon_size),
                speed=self.prey_move_speed)
        else:
            self.prey = AngularPrey(
                self.canvas_shape,
                angle_delta=self.prey_move_angle,
                radius=round(self.radius * self.canvas_width / 2),
                icon_size=(self.icon_size, self.icon_size)
            )

        self.predator = Predator(canvas_size=self.canvas_shape,
                         icon_size=(self.icon_size, self.icon_size),
                         speed=self.predator_move_speed)
        # self.prey = AngularPrey(self.canvas_shape,
        #                         angle_delta=prey_move_angle,
        #                         radius=round(self.radius * self.canvas_width/2),
        #                         icon_size=(self.icon_size, self.icon_size))
        # self.predator = Predator(self.canvas_shape,
        #                          icon_size=(self.icon_size, self.icon_size))
        self.agents = [self.prey, self.predator]

        self._update_obstacles()
        
        # Episode Control variables
        self.trunc_limit = trunc_limit
        self.timesteps = 0
        
        # Learning/Rewards Variables
        self.dist_mult = dist_mult
        self.reward_mult = reward_mult

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

        # # Build a canvas
        # self.canvas_shape = resolution
        # self.canvas = np.ones(self.canvas_shape)
        
        # # Define action space (4 directions + stationary)
        # self.action_space = spaces.Discrete(5,)
        
        # # Define observation space (xy for prey and predator and distance)
        # self.observation_space = spaces.Box(low = np.zeros())

    def _update_obstacles(self):
        for agent in self.agents:
            # Add obstacles
            agent.add_obstacle(self.obstacle_list)
            # Add other agents
            agent.add_obstacle([ag for ag in self.agents if ag != agent])

    def _populate_obstacles(self):
        line = LineSegment(Point(0.3 * self.canvas_width, 0.6 * self.canvas_width),
                           Point(0.7 * self.canvas_width, 0.2 * self.canvas_width))
        line2 = LineSegment(Point(0.5 * self.canvas_width, 0.5 * self.canvas_width),
                            Point(0.8 * self.canvas_width, 0.8 * self.canvas_width))
        for obs in (line, line2):
            boxed_obs = convert_line_to_box(obs, self.canvas_width * 0.01)
            self.obstacle_list.extend(boxed_obs)

    def reset(self, seed=None):
        
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
            # if self.random_prey:
            #     self.randomise_prey_position()
            # if self.random_predator:
            #     self.randomise_predator_position()

            distance = self.calculate_distance(normalise=True)

            if not self.detect_collision() and distance > self.min_distance:
                if self.verbose == 3:
                    print(f"{distance:.3f}")
                elif self.verbose == 4:
                    distance = self.calculate_distance(normalise=False)
                    print(f"{distance:.1f}")
                break

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
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
        
        # Move prey
        self.prey.move(self.prey.sample_action())
        
        # Move Predator
        self.predator.move(action)
        
        # Calculate reward
        reward = self.dist_mult * self.calculate_reward()
        
        # Observation before termination
        obs = self.get_observation()

        # Updates canvas
        self.draw_canvas()

        ## Reset episode if termination conditions met
        # Check for collision
        if self.detect_collision():
            # self.reset()
            reward = 1.0 * self.reward_mult
            done = True
            info["is_success"] = True
        
        # Check if Number of Steps exceed Truncation Limit
        self.timesteps += 1
        if self.timesteps >= self.trunc_limit:
            # self.reset()
            truncated = True
            info["is_success"] = False
    
        return obs, reward, done, truncated, info
    
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
            
    def convert_action(self, action):
        """
        Converts scalar action into (x,y) directional movement.
        
        0: (0, 0)
        1: (0, 1) # Up
        2: (-1, 0) # left
        3: (0, -1) # Down
        4: (1, 0) # Right

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        """
        if action > 0:
            x = np.cos(action * np.pi/2).astype('int')
            y = np.sin(action * np.pi/2).astype('int')
        else:
            x,y = 0,0
        return x,y
    
    def detect_collision(self, object1: Mover=None, object2: Mover=None) -> bool:
        """
        Detect whether the Predator and AngularPrey drone are in contact with each other (overlapping).

        Returns
        -------
        bool
            Collided or not.

        """
        if object1 is None:
            object1 = self.predator
        if object2 is None:
            object2 = self.prey

        x_collided = np.abs(object1.x - object2.x) <= (object1.icon_w/2 + object2.icon_w/2)
        y_collided = np.abs(object1.y - object2.y) <= (object1.icon_h/2 + object2.icon_h/2)
        
        return x_collided & y_collided
    
    def calculate_distance(self,
                           object1: Mover=None, object2: Mover=None,
                           normalise: bool=False) -> float:
        """
        Calculate current distance between Predator and AngularPrey.

        Parameters
        ----------
        normalise : bool, optional
            Whether to scale the position between 0 and 1 by the canvas shape. The default is False.

        Returns
        -------
        float
            Euclidean Distance between Predator and AngularPrey.

        """
        if object1 is None:
            object1 = self.predator
        if object2 is None:
            object2 = self.prey

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
    
    def get_observation(self) -> np.ndarray:
        """
        Obtain observation at current time step.

        Returns
        -------
        obs : np.ndarray
            2D Array if obs_image, else (5,) 1D array.

        """
        if self.obs_image:
            obs = self.canvas.canvas
            return obs
        else:
            # Update position of opposite agent in obstacle
            # self.predator.update_obstacle(len(self.obstacle_list), self.prey)

            # Radial raycasting to obtain distances and points of contact
            obs, rays, obj_types = self.predator.radial_raycast(
                [*self.predator.obstacle_list, self.prey], self.canvas,
                return_rays=self.show_rays,
                num_rays=self.num_rays
            )

            collision_dict = {
                "obstacle": 0,
                "prey": 1,
                "predator": -1
            }

            # Convert and Normalise values
            obs = obs / self.max_width
            obj_types = np.vectorize(collision_dict.get)(obj_types)
            obs = np.r_[obs, obj_types]

            if self.show_rays:
                self.rays = rays

            # pred_pos = self.predator.get_position() / np.array(self.canvas_shape)
            # prey_pos = self.prey.get_position() / np.array(self.canvas_shape)
            #
            # dist = np.linalg.norm(pred_pos - prey_pos)
            #
            # obs = np.r_[pred_pos, prey_pos, dist]
            
            return obs

    def set_frame_delay(self, frame_delay):
        self.frame_delay = frame_delay
