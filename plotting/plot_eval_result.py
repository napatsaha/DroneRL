import os
import string
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import seaborn as sns

import utils
from utils import env as env_utils
from utils.tools import read_run_level
from utils.plot_utils import draw_background


def plot_info_background(
        parent_dir,
        run_base_name,
        run_ids,
        *,
        save=False,
        show=True,
        default_id: int = 0,
        agent_type: Literal['predator', 'prey'] = "prey",
        palette="Set1",
        alphabetical_category=False
):
    """
    Plot informative plot which shows the *spawn area* of either *predator* or *prey*, for various
    run scenarios.
    Accepts spawn data as a 2D vector of [[x0, y0], [x1, y1]], where (x0, y0) is the bottom left corner,
    and (x1, y1) is the top right corner of the spawn area.
    Currently only support either a single spawn point, or a rectangular spawn area:
    - For a single-point spawn (e.g. [[0.4, 0.6], [0.4, 0.6]]), will plot as a slightly enlarged dot.
    - For a variable xy spawn area (e.g. [[0, 0], [0.5, 0.5]]), will plot as a box shape area.

    Parameters
    ----------
    parent_dir :
    run_base_name :
    run_ids :
    save : bool
        Whether to save plot to disk
    show : bool
        Whether to visually display the plot
    default_id : int
        Index of run ID (from run_ids) to use as a basis for plotting background (obstacles).
    agent_type : Literal['predator', 'prey'] (default: prey)
        Which agent to plot the spawn areas
    palette : str
        matplotlib colormap / seaborn color palette name for coloration to distinguish each run.
    alphabetical_category :
        Whether to label each run as A,B,C etc or keep the run name as is.

    Returns
    -------
    None

    """
    cmap = plt.get_cmap(palette)

    # Init plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot base canvas
    default_id = run_ids[default_id]
    canvas = draw_background(parent_dir, run_base_name, default_id)
    ax.imshow(canvas, cmap=plt.get_cmap("gray"), extent=(0, 1, 0, 1), origin="lower")

    for i, run_id in enumerate(run_ids):
        # Get spawn area data from corresponding config file
        config = utils.config.get_config(parent_dir, run_base_name, run_id)
        spawn_area = np.array(config.get("environment").get(f"{agent_type}_spawn_area"))
        fixed_spawn = spawn_area[0,0] == spawn_area[1,0] and \
            spawn_area[0,1] == spawn_area[1,1]

        # Define label if ABC
        if alphabetical_category:
            label = string.ascii_uppercase[i]
        else:
            label = f"{run_base_name}_{run_id}"

        # Use config to plot spawn box
        if not fixed_spawn:
            # Rectangular Spawn
            rect = mpatch.Rectangle(spawn_area[0],
                                    width=spawn_area[1,0]-spawn_area[0,0],
                                    height=spawn_area[1,1]-spawn_area[0,1],
                                    ec=cmap(i), fc=cmap(i, 0.1))
            ax.add_patch(rect)
            ax.annotate(label,
                        (1.1, 0.0), xycoords=rect, ha="left", va="top",
                        color=cmap(i), fontsize=12)
        elif fixed_spawn:
            # Single Point spawn
            dot = mpatch.Circle(spawn_area[0], radius=0.01, fc=cmap(i, 0.8), ec="black")
            ax.add_patch(dot)
            ax.annotate(label, spawn_area[0],
                        (5, 5), textcoords="offset points", color=cmap(i), fontsize=12)

    # Decorations
    fig.suptitle(f"{agent_type.upper()} Spawn Area" +
                 f"{run_base_name} | {run_ids}")
    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        plotpath = os.path.join("plot", parent_dir, f"Eval-Result_{run_base_name}_{'-'.join([str(id) for id in run_ids])}_spawn-plot")
        fig.savefig(plotpath)

    if show:
        plt.show()


def plot_eval_result(
        parent_dir,
        run_base_name,
        run_ids,
        *,
        metric=None,
        save=False,
        show=True,
        max_reps=None,
        palette="Set1",
        alphabetical_category=False
) -> pd.DataFrame:
    """
    Create a boxplot + stripplot of evaluation metrics across different run scenarios.
    If no metric are passed in, will use all available metrics in eval_result.csv
    Number of metrics can vary based on number of agents (e.g. predator1_reward)

    Parameters
    ----------
    parent_dir :
    run_base_name :
    run_ids : List[int]
        Run Scenarios to compare
    metric :
        Evaluation metric to plot.
        If str, plot a single evaluation metric
        If list, plot all items contained in the list
        If None (default), will use and plot every metric available in the eval_result file
    save : bool
        Whether to save plot to disk
    show : bool
        Whether to visually display the plot
    max_reps : int or None
        Max number of Replication (e.g. DQN_*) to read from each Run scenario, to plot.
        If None (default), will read all reps available in each runs (number may differ)
    palette : str
        matplotlib colormap / seaborn color palette name to distinguish each run.
    alphabetical_category :
        Whether to label each run as A,B,C etc or keep the run name as is.

    Returns
    -------
    result : DataFrame
        Collated evaluation result from each run, rep

    """
    # Default values
    base_dir = "logs"
    rep_base_name = "DQN"
    file_base_name = "eval_result.csv"

    # Gather eval result data
    result = read_run_level(
        base_dir, parent_dir, run_base_name, run_ids, rep_base_name, file_base_name,
        index_col=0, max_reps=max_reps
    )

    # Rep-level average for each metric
    result = result.groupby(level=["run", "rep"]).mean()

    # Select appropriate metric to use
    if metric is None:
        metric = result.columns[~result.columns.str.match(r".*[xy]$")]
    elif isinstance(metric, str):
        metric = [metric]

    # Prepare data
    result.reset_index(level="run", inplace=True)
    if alphabetical_category:
        id_mapper = {run_id: string.ascii_uppercase[i] for i, run_id in enumerate(run_ids)}
        result.run = result.run.map(id_mapper)

    # Init Plotting
    fig = plt.figure(figsize=(10, 10))

    for i, scalar in enumerate(metric, start=1):
        # Create new axis
        ax = plt.subplot(2, 2, i)

        # Boxplot and Stripplot
        sns.boxplot(x="run", y=scalar, data=result, ax=ax, hue="run", palette=palette, dodge=False)
        sns.stripplot(x="run", y=scalar, data=result, ax=ax, alpha=0.8, jitter=True, color="grey")

        # Decorations
        ax.set_title(scalar.replace("_", " ").title())
        ax.set_ylabel("")
        ax.legend().set_visible(False)

    # Figure Decorations
    fig.suptitle(f"Evaluation Result\n{run_base_name} | {run_ids}")

    if save:
        plotpath = os.path.join("plot", parent_dir, f"Eval-Result_{run_base_name}_{'-'.join([str(id) for id in run_ids])}_box-plot")
        fig.savefig(plotpath)

    if show:
        plt.show()

    return result


if __name__ == "__main__":
    # Metadata
    parent_dir = "test2"
    run_base_name = "TestWorked"
    run_ids = [1,3,4,5,8]
    max_reps = None
    metric = None
    save = True
    show = True
    palette = "Set1"

    plot_info_background(
        parent_dir,
        run_base_name,
        run_ids,
        save=save,
        show=show,
        palette=palette,
        alphabetical_category=True
    )

    result = plot_eval_result(
        parent_dir,
        run_base_name,
        run_ids,
        max_reps=max_reps,
        show=show,
        save=save,
        palette=palette,
        alphabetical_category=True
    )

    # Show number of reps for each run
    print(result.groupby(by='run').count())
