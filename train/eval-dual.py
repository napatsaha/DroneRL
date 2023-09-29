
import gymnasium as gym
import numpy as np
import os

from algorithm.agent import DualAgent
from envs.dual import DualDrone

parent_dir = "dual1"
run_base_name = "DualDrone"
run_id = 7
run_name = f"{run_base_name}_{run_id}"
rep_name = "DQN_1"

env = DualDrone(trunc_limit=300, predator_move_speed=3,
                frame_delay=10)

agent = DualAgent(env, net_kwargs=dict(net_arch=[64,64]))

agent.load(os.path.join("model", parent_dir, run_name), rep_name)

agent.evaluate(num_eps=10)
