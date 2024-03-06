# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:14:07 2023

@author: napat
"""

from typing import Any, Dict, List, Optional, Type, Tuple, Union

import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import create_mlp

class QNetwork(nn.Module):
    """
    Base Model Network for turning observations into action values.
    
    Handles prediction and weight updates.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete), "Non-Discrete action space not supported for DQN."
        
        if net_arch is None:
            net_arch = [64, 64]
            
        #
        self.observation_space = observation_space
        self.action_space = action_space
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = observation_space.shape[0]

        self.action_dim = int(self.action_space.n)  # number of actions
        q_net = create_mlp(self.features_dim, self.action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)
        self.dtype = q_net[0].weight.dtype
    
    def forward(self, obs: th.Tensor) -> th.Tensor:
        # Torch Tensor check
        if not isinstance(obs, th.Tensor):
            obs = th.tensor(obs)
        # dtype check
        if obs.dtype is not self.dtype:
            obs = obs.type(self.dtype)
        # Check same device
        # if obs.device != self.device:
        #     obs = obs.to(self.device)
        return self.q_net(obs)
    
    def predict(self, observation: th.Tensor, deterministic: bool = True,
                return_output: bool = False) -> Union[th.Tensor, Tuple[th.Tensor, ...]]:
        q_values = self(observation)

        if deterministic:
            # Greedy action
            action = q_values.argmax(dim=-1).reshape(-1)
        else:
            # Softmax probabilistic
            action = th.multinomial(th.softmax(q_values, dim=-1), 1)
        # print(q_values, action)
        if return_output:
            return action, q_values
        else:
            return action
    