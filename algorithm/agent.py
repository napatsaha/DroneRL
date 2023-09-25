# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:22:37 2023

@author: napat
"""
import os
from tqdm.rich import tqdm
import torch as th
import gymnasium as gym
from algorithm.policy import DQNPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.logger import Logger


class DQNAgent:
    """
    Simple learner where the environment is tied to one agent.
    """
    def __init__(
            self,
            env: GymEnv,
            # num_reps: int = 1,
            learning_starts: int = 50000,
            **policy_kwargs
    ):
        # self.num_reps = num_reps
        self.learning_starts = learning_starts

        self.policy = DQNPolicy(
            env.observation_space, env.action_space,
            **policy_kwargs
        )
        self.env = env

    def set_logger(self, logger):
        self.policy.set_logger(logger)

    def learn(
            self,
            total_timesteps: int = 100000,
            log_interval: int = 4,
            progress_bar: bool = False
    ):
        self.policy.setup_learn(total_timesteps, log_interval)
        if progress_bar:
            pbar = tqdm(total=total_timesteps - self.policy.num_timesteps)

        while not self.policy.done:
            eps_reward = []
            state, _ = self.env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                action = self.policy.predict(state)
                nextstate, reward, done, truncated, info = self.env.step(action)
                self.policy.store_transition(state, nextstate, action, reward, done, truncated, info)

                # Perform weight update if conditions met
                if self.policy.num_timesteps > self.learning_starts and \
                        self.policy.num_timesteps % self.policy.train_freq == 0:
                    self.policy.train()

                # Update exploration rates within policy
                self.policy.step()

                if progress_bar:
                    pbar.update()

                state = nextstate
                eps_reward.append(reward)

        self.policy.logger.close()
        self.env.close()
        if progress_bar:
            pbar.refresh()
            pbar.close()

    def save(self, dir_path, run_name):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        th.save(self.policy.q_net.state_dict(),
                os.path.join(dir_path, f"{run_name}.pt"))

    def load(self, path):
        state_dict = th.load(path)
        self.policy.load_state_dict(state_dict)
