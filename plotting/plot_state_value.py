"""
Plot contour maps of trained models

Contour maps show maximum Q-values for each point on state space (2D only),
and greedy action taken according to those Q-values.

Assume fixed position of Prey, and Predator taking only global location observation.
"""

import os
import re
from typing import Tuple, Literal

import numpy as np
import yaml
import torch as th
import matplotlib.pyplot as plt

from algorithm import ALG_DICT
from envs import ENV_DICT
from utils.plot_utils import ACTION_DICT_CARDINAL


# Configurations
def plot_state_value(
        parent_dir: str,
        run_base_name: str,
        run_id: int,
        rep_name: str,
        timestep: str | int | None = None,
        *,
        save: bool = True,
        show: bool = True,
        show_gradient: bool = True,
        prey_spawn_pos: Tuple[float, float] = (0.5, 0.5),
        no_timestep_behaviour: Literal['all', 'final'],
        verbose: int = 2
):
    """

    Parameters
    ----------
    parent_dir :
    run_base_name :
    run_id :
    rep_name :
    timestep :
    save :
    show :
    show_gradient : Show gradient of colours instead of contours for q-values
    prey_spawn_pos : Modifiable spawn position (for both interpolation and visualising
    no_timestep_behaviour : Literal['all', 'final']
        Which model to plot when timestep=None:
        - 'all' -- plot all models with this given rep
        - 'final' -- only plot the final model
    verbose :

    Returns
    -------

    """

    # Download Meta data
    run_name = f"{run_base_name}_{run_id}"
    config_file = os.path.join("config", parent_dir, f"{run_name}.yaml")
    with open(config_file, "r") as file:
        config = yaml.load(file, yaml.SafeLoader)

    AgentClass = ALG_DICT[config["agent_class"]]
    EnvironmentClass = ENV_DICT[config["environment_class"]]

    # Set up folder to save plots
    if save:
        plot_path = os.path.join("plot", parent_dir)
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        plot_path = os.path.join("plot", parent_dir, run_name)
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
    else:
        plot_path = None

    # Setup environment for manually setting prey, and drawing without predators
    env = EnvironmentClass(**config["environment"])
    env.reset()
    env.agents[0].set_position(*np.array(prey_spawn_pos) * env.canvas_width)
    # Draw only prey and obstacles
    env.draw_canvas()
    env.canvas.clear()
    for obstacle in env.obstacle_list:
        env.canvas.draw(obstacle)
    env.canvas.draw(env.agents[0])
    canvas = env.canvas.canvas

    # Setup agent
    agent = AgentClass(env, **config["agent"])

    # Loop over timestep models
    model_file = os.path.join("model", parent_dir, run_name)

    timestep_list = [r.group(1) for r in [re.search("_(\d+).pt", m) for m in os.listdir(model_file)] if r is not None]
    timestep_list = np.unique(timestep_list)

    if timestep is None:
        if no_timestep_behaviour == "all":
            iter_list = np.r_[timestep_list, None]
        elif no_timestep_behaviour == "final":
            iter_list = [timestep]
    else:
        if isinstance(timestep, float):
            timestep = str(int(timestep))
        iter_list = [timestep]

    # learning_starts = config["agent"]['learning_starts']
    # save_interval = config['learn']['save_interval']
    total_timesteps = config['learn']['total_timesteps']
    max_digits = len(str(total_timesteps))

    # timestep_list = np.arange(learning_starts+save_interval, total_timesteps+1, save_interval)

    for timestep in iter_list:
        if timestep is not None and len(str(timestep)) < max_digits:
            timestep = f"{timestep:>0{max_digits}}"

        # Download model
        try:
            agent.load(model_file, rep_name, timestep, verbose=verbose)
        except:
            continue

        # labels for titling and file name
        if timestep is None:
            timestep_label = "Final"
        else:
            timestep_label = str(timestep)
        model_name = " | ".join([run_name, rep_name, timestep_label])

        # Extrapolate data
        policy = agent.agents['predator1']
        policy.q_net = policy.q_net.cpu()

        # Meshgrid containing location details for interpolating q-values
        xx = th.arange(0, 1, 0.02)
        yy = xx

        X, Y = th.meshgrid(xx, yy, indexing='xy')
        XY = th.stack([X, Y], dim=2)

        # Custom prey position as part of q-net input
        prey_pos = th.tensor(prey_spawn_pos).expand(50, 50, -1)

        # Interpolate Q-values
        with th.no_grad():
            inp = th.cat([prey_pos, XY], dim=2)
            q_values = policy.q_net(inp)

            Z_qmax = q_values.max(dim=2)[0]
            # Z_qmean = q_values.mean(dim=2)
            # Z_var = q_values.var(dim=2)
            Z_action = q_values.argmax(dim=2)

        # Init subplots
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Plot Max Q-values
        if show_gradient:
            ax[0].imshow(canvas.T, extent=(0, 1, 0, 1), cmap=plt.get_cmap("gray"), origin="lower")
            ctr = ax[0].imshow(Z_qmax, extent=(0, 1, 0, 1), alpha=0.5)
        else:
            ctr = ax[0].contourf(X, Y, Z_qmax, alpha=0.5)
            ax[0].imshow(canvas.T, extent=(0, 1, 0, 1), cmap=plt.get_cmap("gray"), origin="lower")
        plt.colorbar(ctr, ax=ax[0])
        ax[0].set_title("Max Q-Value" + "\n" + model_name)
        ax[0].axis("off")

        formatter = plt.FuncFormatter(lambda val, pos: ACTION_DICT_CARDINAL[val])
        cmap = plt.get_cmap("viridis", 5)

        # Plot greedy actions taken
        ax[1].imshow(canvas.T, extent=(0, 1, 0, 1), cmap=plt.get_cmap("gray"), origin="lower")
        ctr = ax[1].contourf(X, Y, Z_action, cmap=cmap, alpha=0.5)
        plt.colorbar(ctr, ticks=[0, 1, 2, 3, 4], format=formatter, ax=ax[1])
        ax[1].set_title("Greedy action" + "\n" + model_name)
        ctr.set_clim(-0.5, 4.5)
        ax[1].axis("off")

        # Save and display
        if save:
            plot_name = os.path.join(plot_path, f"{run_name}_{rep_name}_{timestep_label}.png")
            if verbose >= 2:
                print(f"Saving plot to {plot_name}")
            plt.savefig(plot_name)

        if show:
            if verbose >= 2:
                print(f"Showing plot for {model_name}")
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    # Configurations
    parent_dir = "test2"
    run_base_name = "TestWorked"
    run_id = 1
    rep_name = "DQN_1"
    # timestep = "140000"
    timestep = None
    save = False
    show = True

    plot_state_value(parent_dir,
                     run_base_name,
                     run_id,
                     rep_name,
                     timestep,
                     save=save,
                     show=show)
