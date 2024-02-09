
import gymnasium as gym
import numpy as np
import os, yaml

from algorithm import DualAgent, MultiAgent, DQNAgent, ALG_DICT
from envs import DualDrone, DualLateral, MultiDrone, ENV_DICT
from envs.environment_v1 import DroneCatch

# AgentClass = DQNAgent
# EnvironmentClass = MultiDrone

env, agent = None, None

def eval(parent_dir, run_base_name, run_id, rep_name, timestep = None,
         num_eps = 10, frame_delay = 20,
         probabilistic = None,
         **kwargs):
    global env, agent

    run_name = f"{run_base_name}_{run_id}"
    config_file = os.path.join("config", parent_dir, f"{run_name}.yaml")
    with open(config_file, "r") as file:
        config = yaml.load(file, yaml.SafeLoader)

    AgentClass = ALG_DICT[config["agent_class"]]
    EnvironmentClass = ENV_DICT[config["environment_class"]]

    config["environment"].update(kwargs)

    # config["environment"]["min_distance"] = 0.5
    env = EnvironmentClass(**config["environment"])
    # env.trunc_limit = 300

    agent = AgentClass(env, **config["agent"])

    model_file = os.path.join("model", parent_dir, run_name)
    agent.load(model_file, rep_name, timestep)

    if probabilistic is not None:
        agent.probabilistic = probabilistic

    agent.evaluate(num_eps=num_eps, frame_delay=frame_delay)

if __name__ == "__main__":
    parent_dir = "test1"
    run_base_name = "TestAlgo"
    run_id = 4
    rep_name = "DQN_1"
    timestep = None

    eval(parent_dir, run_base_name, run_id, rep_name, timestep,
         frame_delay=5, num_eps=10, trunc_limit=300,
         # obstacle_file="test_codes/obstacle-letterL.csv",
         # show_rays=True, diagnostic=True
         random_action=True
         )
