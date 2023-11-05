from .environment import DroneCatch
from .dual import DualDrone, DualLateral
from .mountain_car import MountainCarWrapper
from .multi import MultiDrone

ENV_DICT = {
    "DroneCatch" : DroneCatch,
    "DualDrone": DualDrone,
    "DualLateral": DualLateral,
    "MultiDrone": MultiDrone
}