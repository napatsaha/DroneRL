
import gymnasium as gym
import numpy as np
import os, yaml

from algorithm.agent import DualAgent
from envs.dual import DualDrone

parent_dir = "dual1"
run_base_name = "DualDrone"
run_id = 9
rep_name = "DQN_1"

run_name = f"{run_base_name}_{run_id}"
config_file = os.path.join("config", parent_dir, f"{run_name}.yaml")
with open(config_file, "r") as file:
    config = yaml.load(file, yaml.SafeLoader)

env = DualDrone(trunc_limit=300, predator_move_speed=3,
                frame_delay=20)

agent = DualAgent(env, net_kwargs=dict(net_arch=[64,64]))

model_file = os.path.join("model", parent_dir, run_name)
agent.load(model_file, rep_name)

agent.evaluate(num_eps=10)
