
from stable_baselines3.dqn import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import numpy as np
import os

run_name = "MountainCar_4"
dir_name = "test2"

agent = DQN.load(os.path.join("model",dir_name,run_name))

env = gym.make("MountainCar-v0", render_mode="human")
# env = MountainCarWrapper(env)
agent.set_env(env)
evaluate_policy(agent, agent.get_env(), n_eval_episodes=5, render=True)
env.close()