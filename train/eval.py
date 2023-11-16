
import gymnasium as gym
import numpy as np
import os, yaml

from algorithm import DualAgent, MultiAgent, DQNAgent, ALG_DICT
from envs import DualDrone, DualLateral, MultiDrone, ENV_DICT
from envs.environment_v1 import DroneCatch

# AgentClass = DQNAgent
# EnvironmentClass = MultiDrone


def eval(parent_dir, run_base_name, run_id, rep_name,
         num_eps = 10, frame_delay = 20,
         probabilistic = None,
         **kwargs):

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
    agent.load(model_file, rep_name)

    if probabilistic is not None:
        agent.probabilistic = probabilistic

    agent.evaluate(num_eps=num_eps, frame_delay=frame_delay)

if __name__ == "__main__":
    parent_dir = "colli1"
    run_base_name = "PredRay"
    run_id = 5
    rep_name = "DQN_1"

    eval(parent_dir, run_base_name, run_id, rep_name,
         frame_delay=10, num_eps=2, #min_distance = 0.2,
         # probabilistic=True,
         show_rays=True)
