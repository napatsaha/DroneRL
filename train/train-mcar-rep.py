
from stable_baselines3.dqn import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_latest_run_id
import gymnasium as gym
import numpy as np
import os

from envs.mountain_car import MountainCarWrapper

num_reps = 10

# Allows previous folders to be added
specific_run_name = "MountainCar_8"
continue_rep = True

parent_dir = "test1"
run_name = "MountainCar"

if specific_run_name is None and run_name is not None:
    current_id = get_latest_run_id(os.path.join("../test_codes/logs", parent_dir), run_name)
    if not continue_rep:
        current_id += 1
    current_dir = f"{run_name}_{current_id}"
else:
    current_dir = specific_run_name

run_dir = os.path.join(parent_dir, current_dir) # "test1/MountainCar_7"

# Create new folders
if not continue_rep:
    os.mkdir(os.path.join("../test_codes/logs", run_dir))
    os.mkdir(os.path.join("model", run_dir))

rep_base_name = "DQN"

# Enables continued training in same directory
for rep in range(num_reps):
    rep_id = get_latest_run_id(os.path.join("../test_codes/logs", run_dir), rep_base_name) + 1
    rep_name = f"{rep_base_name}_{rep_id}"
    rep_path = os.path.join(run_dir, rep_name)

    env = gym.make("MountainCar-v0")
    # env = MountainCarWrapper(env, include_velocity=False)
    # run_name = "MountainCar_7"

    agent = DQN(
        "MlpPolicy", env,
        train_freq=16,
        gradient_steps=8,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        target_update_interval=600,
        learning_starts=10000,
        buffer_size=100000,
        batch_size=128,
        learning_rate=4e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1
    )

    log_dir = os.path.join("../test_codes/logs", rep_path)
    print(f"Logging to {log_dir}")
    logger = configure(
        log_dir,
        format_strings=["csv"]
    )
    agent.set_logger(logger)

    agent.learn(
        total_timesteps=int(2e5),
        log_interval=10,
        progress_bar=True
    )

    model_dir = os.path.join("model",rep_path)
    print(f"Saving model to {model_dir}")
    agent.save(model_dir)

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
