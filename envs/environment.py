# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:52:20 2023

@author: napat
"""

import gymnasium as gym
from gymnasium import Env, Space, spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt

from envs.display import Predator, Prey

class DroneCatch(Env):
    """
    Predator-Prey Drone gym environment for reinforcement learning.
    
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, obs_image: bool=False,
                 resolution: int=800, icon_scale: float=0.1,
                 prey_move_angle: int=5, predator_move_speed: int=5, radius: float=0.8,
                 random_prey: bool=True, random_predator: bool=True,
                 dist_mult: float=0.1, reward_mult: float=1.0,
                 trunc_limit: int=100, frame_delay: int=50,
                 render_mode: str="human", manual_control: bool=False):
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
            Amount of angles (in degrees) that the prey will move in each timestep (counter-clockwise). The default is 5.
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
        
        # Build a canvas
        
        self.resolution = resolution if not isinstance(resolution, tuple) else resolution[0]
        self.canvas_shape = np.array((resolution, resolution))
        self.canvas_width = self.canvas_shape[0]
        self.canvas = np.ones(self.canvas_shape)
        
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
        
        # Predator/Prey configurations
        self.predator_move_speed = predator_move_speed
        self.prey_move_angle = prey_move_angle
        self.radius = radius
        self.random_prey = random_prey
        self.random_predator = random_predator
        
        # Initialises Predator and Prey classes
        self.prey = Prey(self.canvas_shape, 
                         angle_delta=prey_move_angle, 
                         radius=round(self.radius * self.canvas_width/2),
                         icon_size=(self.icon_size, self.icon_size))
        self.predator = Predator(self.canvas_shape,
                                 icon_size=(self.icon_size, self.icon_size))
        
        # Episode Control variables
        self.trunc_limit = trunc_limit
        self.trunc_count = 0
        
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
        
    def reset(self, seed=None):
        
        # Reset Positions for Predator and Prey
        for elem in [self.predator, self.prey]:
            elem.reset_position()
            
        # Randomise if necessary
        if self.random_prey:
            self.randomise_prey_position()
        if self.random_predator:
            self.randomise_predator_position() 

            
        # Imprints new positions onto environment canvas
        self.draw_canvas()
        
        obs = self.get_observation()
            
        self.trunc_count = 0
        
        return obs, {}
        
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take an action and transition the environment to the next step.
        
        Predator moves by the provided action; Prey moves in a constant trajectory 
        around a circle counter-clockwise.

        Parameters
        ----------
        action : int
            Direction to move the Predator (range [0,4]).

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
        self.prey.move_in_circle()
        
        # Move Predator
        delta = np.array([*self.convert_action(action)]) * self.move_speed
        self.predator.move(*delta)
        
        # Updates canvas
        self.draw_canvas()
        
        # Calculate reward
        reward = self.dist_mult * self.calculate_reward()
        
        # Observation before termination
        obs = self.get_observation()
        
        ## Reset episode if termination conditions met
        # Check for collision
        if self.detect_collision():
            # self.reset()
            reward = 1.0 * self.reward_mult
            done = True
            info["is_success"] = True
        
        # Check if Number of Steps exceed Truncation Limit
        self.trunc_count += 1
        if self.trunc_count >= self.trunc_limit:
            # self.reset()
            truncated = True
            info["is_success"] = False
    
        return obs, reward, done, truncated, info
    
    def render(self):
        
        if self.render_mode == "human" and not self.manual_control:
            cv2.imshow("Environment", self.canvas)
            cv2.waitKey(self.frame_delay)
            return None
        elif self.render_mode == "human" and self.manual_control:
            cv2.imshow("Environment", self.canvas)
            key = cv2.waitKeyEx(0)
            action = self.key_dict[key]
            return action
        elif self.render_mode == "rgb_array":
            return self.canvas
    
    def close(self):
        cv2.destroyAllWindows()
    
    def draw_icon(self, canvas, icon, x, y):
        shapeX, shapeY = icon.shape
        canvas[
            round(self.canvas_width - y - 0.5*shapeY):round(self.canvas_width - y + 0.5*shapeY),
            round(x - 0.5*shapeX):round(x + 0.5*shapeX) 
               ] = icon
        return canvas
    
    def draw_canvas(self):
        """
        Draw a frame of canvas based on active elements.

        Returns
        -------
        None.

        """
        # Initialise canvas
        self.canvas = np.ones(self.canvas_shape)
        
        # Draw icons
        for icon in [self.prey, self.predator]:
            self.canvas = self.draw_icon(
                self.canvas, icon.icon, icon.x, icon.y)
            
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
    
    def detect_collision(self) -> bool:
        """
        Detect whether the Predator and Prey drone are in contact with each other (overlapping).        

        Returns
        -------
        bool
            Collided or not.

        """
        x_collided = np.abs(self.predator.x - self.prey.x) <= (self.predator.icon_w/2 + self.prey.icon_w/2)
        y_collided = np.abs(self.predator.y - self.prey.y) <= (self.predator.icon_h/2 + self.prey.icon_h/2)
        
        return x_collided & y_collided
    
    def calculate_distance(self, normalise: bool=False) -> float:
        """
        Calculate current distance between Predator and Prey.

        Parameters
        ----------
        normalise : bool, optional
            Whether to scale the position between 0 and 1 by the canvas shape. The default is False.

        Returns
        -------
        float
            Euclidean Distance between Predator and Prey.

        """
        scale = np.array(self.canvas_shape) if normalise else 1.0
        pred_pos = self.predator.get_position() / scale
        prey_pos = self.prey.get_position() / scale
        
        dist = np.linalg.norm(pred_pos - prey_pos)
        
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
            obs = self.canvas
            return obs
        else:
            pred_pos = self.predator.get_position() / np.array(self.canvas_shape)
            prey_pos = self.prey.get_position() / np.array(self.canvas_shape)
            
            dist = np.linalg.norm(pred_pos - prey_pos)
            
            obs = np.r_[pred_pos, prey_pos, dist]
            
            return obs
        
    def randomise_predator_position(self):
        
        # Loop to ensure no overlap between Predator and Prey
        while True:
            rand_pos = np.random.randint(self.canvas_shape)
            self.predator.reset_position(*rand_pos)
            
            if not self.detect_collision():
                break
            
    def randomise_prey_position(self):
        angle = np.random.randint(360)
        self.prey.reset_position(angle)
        