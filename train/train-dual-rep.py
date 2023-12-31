"""
Train using custom DQN Learner.

Repetition framework based on train-mcar-rep.py
"""

from stable_baselines3.dqn import DQN
from algorithm.agent import DualAgent

from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_latest_run_id
import gymnasium as gym
import numpy as np
import os, yaml

from envs.dual import DualDrone

# Default config file to read from
config_file = "config/dual1/default.yaml"

num_reps = 10

# Allows previous folders to be added
specific_run_name = None
continue_rep = True if specific_run_name is not None else False

parent_dir = "dual1"
run_name = "DualDrone"

if specific_run_name is None and run_name is not None:
    current_id = get_latest_run_id(os.path.join("logs", parent_dir), run_name)
    if not continue_rep:
        current_id += 1
    current_dir = f"{run_name}_{current_id}"
else:
    current_dir = specific_run_name

run_dir = os.path.join(parent_dir, current_dir) # "e.g. test1/MountainCar_7"

# Create new folders
for path in ["logs", "model", "config"]:
    if not os.path.exists(os.path.join(path, parent_dir)):
        os.mkdir(os.path.join(path, parent_dir))
    if path != "config":
        joint_path = os.path.join(path, run_dir)
        if not os.path.exists(joint_path):
            os.mkdir(joint_path)
# if not continue_rep:
#     os.mkdir(os.path.join("logs", run_dir))
#     os.mkdir(os.path.join("model", run_dir))

rep_base_name = "DQN"

# Read config and store a copy for current run
with open(config_file, "r") as file:
    config = yaml.load(file, yaml.SafeLoader)

# Stores new file
new_config_file = os.path.join("config", parent_dir, f"{current_dir}.yaml")
with open(new_config_file, "w") as file:
    yaml.dump(config, file)

# Enables continued training in same directory
for rep in range(num_reps):
    rep_id = get_latest_run_id(os.path.join("logs", run_dir), rep_base_name) + 1
    rep_name = f"{rep_base_name}_{rep_id}"
    rep_path = os.path.join(run_dir, rep_name)

    env = DualDrone(
        **config["environment"]
        # reward_mult=10.0,
        # predator_move_speed=3,
        # trunc_limit=300
    )

    agent = DualAgent(
        env,
        **config["agent"]
        # train_freq=16,
        # gradient_steps=8,
        # gamma=0.99,
        # exploration_fraction=0.2,
        # exploration_final_eps=0.07,
        # target_update_interval=600,
        # learning_starts=10000,
        # buffer_size=100000,
        # batch_size=128,
        # learning_rate=1e-4,
        # net_kwargs=dict(net_arch=[64, 64])
    )

    log_dir = os.path.join("logs", rep_path)
    # print(f"Logging to {log_dir}")
    # logger = configure(
    #     log_dir,
    #     format_strings=["csv"]
    # )
    # agent.set_logger(logger)

    agent.set_logger_by_dir(log_dir, format_strings=['csv'])

    agent.learn(
        # total_timesteps=int(2e5),
        # log_interval=10,
        progress_bar=True,
        **config["learn"]
    )

    model_dir = os.path.join("model",run_dir)

    agent.save(model_dir, rep_name)

# n_eval=30
# ep_rew, ep_len = evaluate_policy(agent, agent.get_env(), n_eval_episodes=n_eval, return_episode_rewards=True)
# print(f"evaluation over {n_eval} episodes:")
# print(f"Mean reward:\t{np.mean(ep_rew):.3f}")
# print(f"Mean length:\t{np.mean(ep_len)}")
#
# ## Rendering (Optional)
# env = gym.make("MountainCar-v0", render_mode="human")
# # env = MountainCarWrapper(env)
# agent.set_env(env)
# evaluate_policy(agent, agent.get_env(), n_eval_episodes=5, render=True)
# env.close()
