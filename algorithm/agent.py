# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:22:37 2023

@author: napat
"""
import os
from typing import Optional, List, Dict
from tqdm.rich import tqdm
import torch as th
import gymnasium as gym
from algorithm.policy import DQNPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.logger import Logger, configure
from collections import namedtuple
from envs import DualDrone, MultiDrone

class DQNAgent:
    """
    Simple learner where the environment is tied to one agent.
    """
    def __init__(
            self,
            env: GymEnv,
            # num_reps: int = 1,
            learning_starts: int = 50000,
            save_interval: int = None,
            **policy_kwargs
    ):
        # self.num_reps = num_reps
        self.learning_starts = learning_starts
        self.save_interval = save_interval

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
        model_path = os.path.join(dir_path, f"{run_name}.pt")
        print(f"Saving model to {model_path}")
        th.save(self.policy.q_net.state_dict(),
                model_path)

    def load(self, dir_path, run_name):
        file = next(filter(lambda x: x.endswith(f"{run_name}.pt"), os.listdir(dir_path)))
        file_path = os.path.join(dir_path, file)
        print(f"Loading model from {file_path}")
        state_dict = th.load(file_path)
        self.policy.q_net.load_state_dict(state_dict)

    def evaluate(self, num_eps: int = 20, render: bool = True,
                 frame_delay: int = 20):

        if render:
            self.env.set_frame_delay(frame_delay)

        for episode in range(num_eps):
            state, _ = self.env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                action = self._predict_action(state)

                nextstate, reward, done, truncated, info = self.env.step(action)

                if render:
                    self.env.render()

                state = nextstate

        self.env.close()

    def _predict_action(self, state):
        action = self.policy.predict(state, deterministic=True)
        return action


class DualAgent:
    """
    Simple learner where the environment is tied to one agent.
    """

    agents: dict[str, DQNPolicy]
    def __init__(
            self,
            env: DualDrone,
            agent_order: Optional[List[str]] = ("predator", "prey"),
            # num_reps: int = 1,
            learning_starts: int = 50000,
            verbose: Optional[int] = 0,
            **policy_kwargs
    ):
        # self.num_reps = num_reps
        self.verbose = verbose
        self.learning_starts = learning_starts
        self.num_timesteps = 0

        self.env = env

        self.agent_order = agent_order
        self.agents = {name: None for name in self.agent_order}

        for i, agent_name in enumerate(self.agent_order):
            agent = DQNPolicy(
                env.observation_space[i],
                env.action_space[i],
                **policy_kwargs
            )
            self.agents[agent_name] = agent

        # self.predator_policy = DQNPolicy(
        #     env.observation_space, env.action_space,
        #     **policy_kwargs
        # )
        #
        # self.prey_policy = DQNPolicy(
        #     env.observation_space, env.action_space,
        #     **policy_kwargs
        # )
        #
        #
        # Agent = namedtuple("Agent", self.agent_order)
        # self.agents = Agent(self.predator_policy, self.prey_policy)


    def set_logger(self, logger):
        if isinstance(logger, str):
            self.set_logger_by_dir(logger)
        else:
            raise Exception("Setting single logger is not supported for multiple agents."
                            "Please pass log_dir to .set_logger_by_dir() instead.")

    def set_logger_by_dir(self, log_dir, format_strings: List[str] = ["csv"]):

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        for agent_name, agent in self.agents.items():
            log_to = os.path.join(log_dir, agent_name)
            print(f"Logging {agent_name} to {log_to}")
            if not os.path.exists(log_to):
                os.mkdir(log_to)
            logger = configure(log_to, format_strings)
            agent.set_logger(logger)

    def learn(
            self,
            total_timesteps: int = 100000,
            log_interval: int = 4,
            save_interval: int = None,
            progress_bar: bool = False,
            dir_path = None,
            run_name = None,
    ):
        # Setup learn in each policy
        self._setup_learn(total_timesteps, log_interval, self.learning_starts)

        # Initialise progress bar
        if progress_bar:
            pbar = tqdm(total=total_timesteps)

        times_model_saved = 0
        max_digits = len(str(total_timesteps))

        # Loop through episodes
        while self.num_timesteps < total_timesteps:
            state, _ = self.env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                start_learning = self.num_timesteps > self.learning_starts

                # Moves randomly until started learning
                action = [
                    policy.predict(state[i], random=not start_learning) for i, policy in enumerate(self.agents.values())
                ]

                nextstate, reward, done, truncated, info = self.env.step(action)

                # For each agent in environment: store, train and step
                for i, policy in enumerate(self.agents.values()):
                    policy.store_transition(state[i], nextstate[i], action[i], reward[i], done, truncated, info)

                    # if self.num_timesteps % log_interval == 0:
                    #     print(self.num_timesteps, state[i], action[i], done, truncated)

                    # Perform weight update if conditions met
                    if start_learning and \
                            policy.num_timesteps % policy.train_freq == 0:
                        policy.train()

                    # Update exploration rates within policy
                    policy.step(start_learning)

                # Save intermediate model at every `save_interval` timesteps
                if start_learning and self.num_timesteps % save_interval == 0:
                    step_name = f"{times_model_saved:02}_{self.num_timesteps:0{max_digits}}"
                    self.save(dir_path, run_name, step_name)
                    times_model_saved += 1

                self.num_timesteps += 1

                if progress_bar:
                    pbar.update()

                state = nextstate

        self.close_logger()
        self.env.close()
        if progress_bar:
            pbar.refresh()
            pbar.close()

    def close_logger(self):
        for policy in self.agents.values():
            policy.logger.close()

    def save(self, dir_path, run_name, step_name: str = None):
        """
        Specify folder (dir_path) to save models and name of the run (run_name).
        For each model in agents, a model will be saved with the name
        \"{run_name}_prey.pt\", \"{run_name}_predator.pt\" etc.

        :param dir_path: Folder to save all models
        :param run_name: Starting name of the file
        :return:
        """
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        for agent_name, policy in self.agents.items():
            file_name = f"{run_name}_{agent_name}_{step_name}.pt" if step_name is not None else \
                f"{run_name}_{agent_name}.pt"
            model_to = os.path.join(dir_path, file_name)
            print(f"Saving {agent_name} model to {model_to}")
            th.save(policy.q_net.state_dict(),
                    model_to)

    def load(self, dir_path, run_name, timestep = None):
        for name, policy in self.agents.items():
            # file = next(filter(lambda x: x.endswith(f"{run_name}_{name}.pt"), os.listdir(dir_path)))
            if timestep is not None:
                file = f"{run_name}_{name}_{timestep}.pt"
            else:
                file = f"{run_name}_{name}.pt"
            file_path = os.path.join(dir_path, file)
            print(f"Loading {name} model from {file_path}")
            state_dict = th.load(file_path)
            policy.q_net.load_state_dict(state_dict)

    def evaluate(self, num_eps: int = 20, render: bool = True,
                 frame_delay: int = 20):

        if render:
            self.env.set_frame_delay(frame_delay)

        for episode in range(num_eps):
            state, _ = self.env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                action = [
                    policy.predict(state[i], deterministic=True)
                    for i, policy in enumerate(self.agents.values())
                ]

                nextstate, reward, done, truncated, info = self.env.step(action)

                if render:
                    self.env.render()

                state = nextstate

        self.env.close()

    def _setup_learn(self, total_timesteps, log_interval, learning_starts):
        for policy in self.agents.values():
            policy.setup_learn(total_timesteps, log_interval, learning_starts)


class MultiAgent(DualAgent):
    def __init__(self, env: MultiDrone, *args, **kwargs):
        super().__init__(env,
                         agent_order=env.agent_list,
                         *args, **kwargs)


if __name__ == "__main__":
    env = DualDrone()
    agent = DualAgent(env)
    # agent.set_logger_by_dir(os.path.join("logs"))