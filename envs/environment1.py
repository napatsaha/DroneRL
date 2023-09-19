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
from envs.environment import DroneCatch

class DualDrone(DroneCatch):
    """
    DroneCatch environment where both Predator and Prey are able to learn simultaneously.
    
    Observations and Rewards returned will have an extra dimension:
        [predator, prey]
        
    Similarly, actions passed to step() must also have an extra dimension:
        [predator, prey]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.action_space = spaces.MultiDiscrete([5,3])
        
        self.observation_space = [self.observation_space for _ in range(2)]
    
    def step(self, action: int):
        
        
        
    
    
    