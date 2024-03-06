# -*- coding: utf-8 -*-
"""
Algorithms and Policies for custom training
DQN

Created on Tue Sep  5 11:05:46 2023

@author: napat
"""
import csv
from _csv import Writer
from io import TextIOWrapper
import time, sys
from typing import Any, Dict, List, Optional, Type, Tuple, Union, Iterable

import os
import numpy as np
import torch as th
from gymnasium import spaces
from collections import deque
from torch.nn import functional as F

from gymnasium import Space

from stable_baselines3.common.buffers import ReplayBuffer
# from stable_baselines3.common.policies import BasePolicy
# from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import polyak_update, get_latest_run_id, safe_mean
# from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.utils import get_device, get_linear_fn, obs_as_tensor
from stable_baselines3.common.logger import configure, Logger

from algorithm.network import QNetwork
from utils.rl_utils import Prediction


class DQNPolicy:
    """
    Deep Q-Network policy that is decoupled from an environment.
    
    Can train with only transition data, allowing outside processes to train
    same environment with multiple agents.
    
    
    
    """
    _episode_num: int
    trajectory_file: TextIOWrapper
    trajectory_writer: Writer

    def __init__(
            self, 
            observation_space: Space,
            action_space: Space,
            name: str = None,
            learning_rate: float = 1e-4,
            buffer_size: int = 1_000_000,
            batch_size: int = 100,
            gamma: float = 0.99,
            tau: float = 1.0,
            train_freq: int = 4,
            gradient_steps: int = -1,
            total_timesteps: int = 100000,
            target_update_interval: int = 10,
            log_interval: int = 4,
            save_log_trajectory: bool = False,
            probabilistic: Union[bool, Tuple[bool], List[bool]] = False,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            stats_window_size: int = 100,
            max_grad_norm: float = 10.0,
            net_kwargs=None,
            optim_kwargs=None,
            # logger: Optional[Logger] = None,
            # log_dir="logs",
            # log_name=None,
            # log_output=["csv","stdout"],
            # reset_num_timesteps: bool = False
    ):

        self.learning_starts = None
        self.algorithm_name = "DQN"
        self.name = name
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.replay_buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = get_device("auto")

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.gradient_steps = gradient_steps

        # Epsilon-softmax combination
        if isinstance(probabilistic, (list, tuple)):
            self._probabilistic_greedy = probabilistic[1] # When 1 - epsilon
            self._probabilistic_random = probabilistic[0] # When epsilon
        else:
            self._probabilistic_greedy = probabilistic
            self._probabilistic_random = probabilistic

        self.exploration_rate = 1.0
        self.exploration_schedule = get_linear_fn(
            exploration_initial_eps, 
            exploration_final_eps,
            exploration_fraction)

        self.episode_done = False
        self.num_timesteps = 0
        self.total_timesteps = total_timesteps
        self.target_update_freq = target_update_interval
        self.log_interval = log_interval
        self.train_freq = train_freq
        self.stats_window_size = stats_window_size
        
        self.net_kwargs = net_kwargs if net_kwargs is not None else {}
        self.optim_kwargs = optim_kwargs if optim_kwargs is not None else {}

        self.logger = None
        self.save_log_trajectory = save_log_trajectory
        self.trajectory_file = None
        self.trajectory_writer = None
        # if logger is not None:
        #     self.logger = logger
        # else:
        #     self.logger = self._setup_logger(log_output, log_dir, log_name, reset_num_timesteps)

        self._build()
        self._setup()
        
        self._n_updates = 0
        self.done = False
        
    def _build(self):
        """Set up model and optimizer."""
        self.q_net = self._make_q_net()
        self.q_net_target = self._make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = th.optim.Adam(
            self.q_net.parameters(),
            lr = self.learning_rate,
            **self.optim_kwargs
            )
                
    def _make_q_net(self) -> QNetwork:
        """Convenient Q Network constructor with device casting."""
        return QNetwork(
            self.observation_space, self.action_space, 
            **self.net_kwargs
        ).to(self.device)
    
    def _setup(self):
        """Setup Replay Buffer."""
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.observation_space, self.action_space)
        self.ep_rew_buffer = deque(maxlen=self.stats_window_size)
        self.ep_len_buffer = deque(maxlen=self.stats_window_size)
        self.rewards: List[float] = []
        self._episode_num = 0
        self.start_time = time.time_ns()
    
    def _setup_logger(self, log_output, log_dir, log_name, reset_num_timesteps) -> Logger:
        """
        Default logger setup if no logger is passed through.
        Depreciated for passing explicit logger
        """
        log_name = self.algorithm_name if log_name is None else log_name
        lastest_id = get_latest_run_id(log_dir, log_name)
        if reset_num_timesteps:
            lastest_id -= 1
        run_name = f"{log_name}_{lastest_id + 1}"
        save_path = os.path.join(log_dir, run_name)
        self.run_name = run_name

        logger = configure(save_path, log_output)
        return logger

    def _update_target_net(self, tau: float=None) -> None:
        """Perform Polyak Update on target Q Network."""
        if tau is None: tau = self.tau
        polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), tau)

    def _update_episode_info(self, reward, done):
        """
        Update info buffers for reward, episode length, etc.

        :param reward:
        :param done:
        :return:
        """
        self.episode_done = done
        self.rewards.append(reward)

        if done:
            # At end of Episode
            self.ep_rew_buffer.append(sum(self.rewards))
            self.ep_len_buffer.append(len(self.rewards))
            self.rewards = []
            self._episode_num += 1

    def _dump_logs(self):

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)

        if len(self.ep_len_buffer) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean(self.ep_rew_buffer))
            self.logger.record("rollout/ep_len_mean", safe_mean(self.ep_len_buffer))

        self.logger.record("rollout/exploration_rate", self.exploration_rate)

        self.logger.record("time/episodes", self._episode_num)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps)

        self.logger.dump(step=self.num_timesteps)
        
    def set_logger(self, logger: Logger):
        self.logger = logger

    def store_transition(self, obs, next_obs, action, reward, done, truncated, info):
        """Store transition into buffer after storing truncated in info"""
        obs = np.array(obs)
        next_obs = np.array(next_obs)
        action = np.array(action)
        reward = np.array(reward)
        done = np.array(done)
        info["TimeLimit.truncated"] = truncated
        infos = [info]

        self._update_episode_info(reward, done | truncated)

        # self.logger.record("reward", reward)

        self.replay_buffer.add(obs, next_obs, action, reward, done, infos)

    def step(self, start_learning: bool = True):
        """Update necessary values after each step in environment.

        Primarily, updates exploration rate, and target Q Network.
        """
        self.num_timesteps += 1

        if start_learning:
            # Updates exploration rate
            self.exploration_rate = self.exploration_schedule(
                1.0 - (self.num_timesteps - self.learning_starts) /
                (self.total_timesteps - self.learning_starts)
            )

            # Updates target network at intervals
            if self.num_timesteps % self.target_update_freq == 0:
                self._update_target_net()

        # Terminate training session
        if self.num_timesteps == self.total_timesteps:
            self.done = True

        # Only record once at the end of episode and when at log_interval intervals
        if self.episode_done and self._episode_num % self.log_interval == 0:
            self._dump_logs()

    def setup_learn(self, total_timesteps, log_interval, learning_starts, log_dir=None):
        if self.logger is None:
            raise Exception("Logger has not been setup yet. Cannot start learning.")

        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.learning_starts = learning_starts

        if self.save_log_trajectory:
            log_file = os.path.join(log_dir, self.name, "trajectory.csv")
            self.trajectory_file = open(log_file, "a+", newline='')
            self.trajectory_writer = csv.writer(self.trajectory_file)

            # Init headers
            if self.save_log_trajectory:
                n_states = self.observation_space.shape[0]
                n_actions = self.action_space.n
                st = [f"s{s}" for s in range(n_states)]
                qt = [f"q{a}" for a in range(n_actions)]
                header = ["episode", "timestep", *st, "action", "reward", *qt, "explore"]
                self.trajectory_writer.writerow(header)

    def finish_learn(self):
        """
        Close any loggers
        """
        self.logger.close()

        if self.save_log_trajectory:
            self.trajectory_file.close()

    def get_qvalues(self, obs: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        """
        Returns q-values for given observation

        Parameters
        ----------
        obs : Union[th.Tensor, np.ndarray]
            observation

        Returns
        -------

        """
        with th.no_grad():
            obs = th.Tensor(obs).to(self.device)
            q = self.q_net(obs)
            return q

    def predict(self, obs: Union[th.Tensor, np.ndarray], *,
                deterministic: bool = False, random: bool = False,
                return_qvalues: bool = False) -> Prediction: #Union[int, tuple[int, th.Tensor]]:
        """
        Make prediction based on epsilon-greedy or softmax exploration.

        Parameters
        ----------
        obs : Union[th.Tensor, np.ndarray]
            observation
        return_qvalues : bool
            Whether to compute qvalues
        random : bool
            Whether to force policy to explore
        deterministic : bool
            Whether to force policy to be greedy

        Returns
        -------
        action : Prediction
            Prediction object that contains information on chosen action, qvalues, and whether the policy explored

        """
        prediction = Prediction()

        # qvalues = None
        explore = np.random.rand() < self.exploration_rate if not random else True

        if not deterministic and explore and not self._probabilistic_random:
            # Uniform random choice
            prediction.action = self.action_space.sample()
            if return_qvalues:
                prediction.qvalues = self.get_qvalues(obs)
        elif not deterministic and \
                ((not explore and self._probabilistic_greedy) or
                 (explore and self._probabilistic_random)):
            # Softmax probabilistic choice
            with th.no_grad():
                prediction.qvalues = self.get_qvalues(obs)
                prediction.action = prediction.softmax_predict()

                # obs = obs_as_tensor(np.array(obs), self.device)
                # if return_qvalues:
                #     action, qvalues = self.q_net.predict(obs, deterministic=False, return_output=return_qvalues)
                # else:
                #     action = self.q_net.predict(obs, deterministic=False)
                # if len(action) == 1:
                #     action = action.item()
        else:
            # Greedy deterministic choice
            with th.no_grad():
                prediction.qvalues = self.get_qvalues(obs)
                prediction.action = prediction.greedy_predict()

                # obs = obs_as_tensor(np.array(obs), self.device)
                # if return_qvalues:
                #     action, qvalues = self.q_net.predict(obs, return_output=return_qvalues)
                # else:
                #     action = self.q_net.predict(obs)
                # if len(action) == 1:
                #     action = action.item()

        prediction.explore = explore

        return prediction

        # if return_qvalues:
        #     return action, qvalues
        # else:
        #     return action
    
    def train(self, gradient_steps: Optional[int] = None,
              batch_size: int = None) -> None:
        """
        Perform Bellman's Equation update as many times as gradient_steps by
        sampling batch_size samples from the replay buffer.

        Parameters
        ----------
        gradient_steps : int, optional
            DESCRIPTION. The default is 4.
        batch_size : int, optional
            The default is 100.

        Returns
        -------
        None
        """
        # Based on:
        # stable_baselines3.dqn.dqn.DQN.train()

        if gradient_steps is None:
            gradient_steps = self.gradient_steps

        if gradient_steps < 0:
            gradient_steps = self.train_freq
        
        if batch_size is None:
            batch_size = self.batch_size

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # print(f"{np.mean(losses):.2f}")
        # Increase update counter
        self._n_updates += gradient_steps
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def log_trajectory(self, episode: int, timestep: int, state: np.ndarray, action: int, reward: float,
                       qvalues: th.Tensor, explore: bool):
        """
        Log trajectory onto the policy's trajectory file in the following order:
        - episode number since start of training
        - timestep since episode start
        - (ns) x observation values at current step
        - action: chosen action according to policy (may include exploration)
        - reward from chosen action
        - (na) x q-values calculated from current observation

        where:
        - ns = number of observations
        - na = number of actions (discrete)

        Parameters
        ----------
        episode : number of current episode
        timestep : timestep since episode start
        state : observation at this timestep
        action : chosen action from training policy at this timestep
        reward : received reward at this timestep

        Returns
        -------
        None

        """
        if self.save_log_trajectory:
            qvalues = qvalues.numpy(force=True)
            explore = int(explore)
            row = [episode, timestep, *state, action, reward, *qvalues, explore]
            self.trajectory_writer.writerow(row)


