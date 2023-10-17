
import gymnasium as gym
import numpy as np
import os, yaml

from algorithm import DualAgent, MultiAgent
from envs import DualDrone, DualLateral, MultiDrone

AgentClass = DualAgent
EnvironmentClass = DualLateral
parent_dir = "dual2"
run_base_name = "DualLateral"
run_id = 10
rep_name = "DQN_1"

run_name = f"{run_base_name}_{run_id}"
config_file = os.path.join("config", parent_dir, f"{run_name}.yaml")
with open(config_file, "r") as file:
    config = yaml.load(file, yaml.SafeLoader)

# config["environment"]["min_distance"] = 0.5
env = EnvironmentClass(**config["environment"])
# env.trunc_limit = 300

agent = AgentClass(env, **config["agent"])

model_file = os.path.join("model", parent_dir, run_name)
agent.load(model_file, rep_name)

agent.evaluate(num_eps=10, frame_delay=50)
