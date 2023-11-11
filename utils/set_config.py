# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:37:29 2023

@author: napat

Pull out and set default config of environment constructors and SB3 Algorithms into yaml file.

Useful to update default parameter when environment class is modified.
"""

import yaml, os
from envs.environment_v1 import DroneCatch
from stable_baselines3 import DQN
from algorithm import DQNAgent, DQNPolicy

# param_list = ['resolution', 'icon_scale', 'prey_move_angle', 'predator_move_speed', 'dist_mult', 'reward_mult', 
#               'trunc_limit', 'frame_delay', 'render_mode', 'radius', 'obs_image']
# env = DroneCatch()
# dct = {key: val for key, val in vars(env).items() if key in param_list}

def pull_default_params(object_class):
    param_defaults = object_class.__init__.__defaults__
    param_names = object_class.__init__.__code__.co_varnames[-len(param_defaults):]
    dct = {key: val for key, val in zip(param_names, param_defaults)}
    return dct

def pull_default_params_func(func):
    param_defaults = func.__defaults__
    param_names = func.__code__.co_varnames[-len(param_defaults):]
    dct = {key: val for key, val in zip(param_names, param_defaults)}
    return dct

if __name__ == "__main__":
    env = DroneCatch
    agent = DQNAgent
    policy = DQNPolicy
    output = "default2.yaml"

    alg_dict = pull_default_params(policy)
    alg_dict.update(pull_default_params(agent))
    dct = {
        "environment_class": f"{env.__name__}-{env.version}",
        "environment": pull_default_params(env),
        "learn": pull_default_params_func(agent.learn),
        "agent_class": agent.__name__,
        "agent": alg_dict,
           }

    with open(os.path.join("config", output), "w") as file:
        yaml.dump(dct, file)