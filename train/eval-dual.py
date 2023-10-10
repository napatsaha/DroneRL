
import gymnasium as gym
import numpy as np
import os, yaml

from algorithm.agent import DualAgent
from envs.dual import DualDrone, DualLateral

parent_dir = "dual2"
run_base_name = "DualLateral"
run_id = 12
rep_name = "DQN_5"

run_name = f"{run_base_name}_{run_id}"
config_file = os.path.join("config", parent_dir, f"{run_name}.yaml")
with open(config_file, "r") as file:
    config = yaml.load(file, yaml.SafeLoader)

# config["environment"]["min_distance"] = 0.5
env = DualLateral(**config["environment"])
env.trunc_limit = 100

agent = DualAgent(env, **config["agent"])

model_file = os.path.join("model", parent_dir, run_name)
agent.load(model_file, rep_name)

agent.evaluate(num_eps=10, frame_delay=10)
