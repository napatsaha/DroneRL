# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:37:29 2023

@author: napat

Pull out and set default config of environment constructors and SB3 Algorithms into yaml file.

Useful to update default parameter when environment class is modified.
"""

import yaml, os
from envs.environment import DroneCatch
from stable_baselines3 import DQN

# param_list = ['resolution', 'icon_scale', 'prey_move_angle', 'predator_move_speed', 'dist_mult', 'reward_mult', 
#               'trunc_limit', 'frame_delay', 'render_mode', 'radius', 'obs_image']
# env = DroneCatch()
# dct = {key: val for key, val in vars(env).items() if key in param_list}

def pull_default_params(object_class):
    param_defaults = object_class.__init__.__defaults__
    param_names = object_class.__init__.__code__.co_varnames[-len(param_defaults):]
    dct = {key: val for key, val in zip(param_names, param_defaults)}
    return dct

if __name__ == "__main__":
    dct = {"environment": pull_default_params(DroneCatch),
           "algorithm": "dqn",
           "learn": {"total_timesteps": 2.0e+5},
           "model": pull_default_params(DQN)}

    with open(os.path.join("config","default.yaml"), "w") as file:
        yaml.dump(dct, file)