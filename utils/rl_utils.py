from typing import Union

import torch as th
import numpy as np


class Prediction:
    """
    Container object to store information about predicted action, qvalues (weights)
    and whether or not action was chosen randomly (from epsilon-greedy policies).
    """
    action: Union[int, list[int]]
    qvalues: Union[th.Tensor, np.ndarray]
    explore: bool

    def __init__(self, action=None, qvalues=None, explore=None):
        self.action = action
        self.qvalues = qvalues
        self.explore = explore

    def softmax_predict(self, q_values=None):
        if q_values is None: q_values = self.qvalues
        action = th.multinomial(th.softmax(q_values, dim=-1), 1)
        if len(action) == 1:
            action = action.item()
        return action

    def greedy_predict(self, q_values=None):
        if q_values is None: q_values = self.qvalues
        action = q_values.argmax(dim=-1).reshape(-1)
        if len(action) == 1:
            action = action.item()
        return action