from .agent import MultiAgent, DualAgent, DQNAgent
from .policy import DQNPolicy
from .network import QNetwork

ALG_DICT = {
    "DQNAgent": DQNAgent,
    "MultiAgent": MultiAgent,
    "DualAgent": DualAgent
}

POLICY_DICT = {
    "DQN": DQNPolicy
}