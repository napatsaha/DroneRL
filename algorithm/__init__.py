from .agent import MultiAgent, DualAgent, DQNAgent
from .policy import DQNPolicy
from .network import QNetwork

ALG_DICT = {
    "MultiAgent": MultiAgent,
    "DualAgent": DualAgent,
    "DQNAgent": DQNAgent
}
