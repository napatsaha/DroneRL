# from .environment_v1 import DroneCatch
from .dual import DualDrone, DualLateral
from .mountain_car import MountainCarWrapper
from .multi import MultiDrone
from . import geometry, environment_v1, environment_v0

ENV_DICT = {
    "DroneCatch-v0": environment_v0.DroneCatch,
    "DroneCatch-v1": environment_v1.DroneCatch,
    "DualDrone": DualDrone,
    "DualLateral": DualLateral,
    "MultiDrone": MultiDrone
}