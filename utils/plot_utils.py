import numpy as np

from utils import env as env_utils
from utils.config import get_config

ACTION_DICT_CARDINAL = \
    {0: "Stationary",
     1: "Up",
     2: "Left",
     3: "Down",
     4: "Right"}


def draw_background(parent_dir, run_base_name, run_id) -> np.ndarray:
    """
    Quick function to draw base background of a certain environment.

    Parameters
    ----------
    parent_dir :
    run_base_name :
    run_id :

    Returns
    -------

    """
    config = get_config(parent_dir, run_base_name, run_id)
    env = env_utils.create_env(config)

    canvas = env.draw_canvas(draw_agents=False, return_canvas=True)
    canvas = canvas.canvas.T
    return canvas