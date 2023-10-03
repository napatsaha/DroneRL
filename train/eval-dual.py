
import gymnasium as gym
import numpy as np
import os, yaml

from algorithm.agent import DualAgent
from envs.dual import DualDrone

parent_dir = "dual1"
run_base_name = "DualDrone"
run_id = 11
rep_name = "DQN_5"

run_name = f"{run_base_name}_{run_id}"
config_file = os.path.join("config", parent_dir, f"{run_name}.yaml")
with open(config_file, "r") as file:
    config = yaml.load(file, yaml.SafeLoader)

env = DualDrone(**config["environment"])

agent = DualAgent(env, **config["agent"])

model_file = os.path.join("model", parent_dir, run_name)
agent.load(model_file, rep_name)

agent.evaluate(num_eps=10)
