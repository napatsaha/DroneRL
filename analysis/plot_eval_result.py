import os
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
        default_id: int = 0,
        agent_type: Literal['predator', 'prey'] = "prey",
        palette="Set1"
):
    """
    Plot informative spawn areas for included `run_ids`

    Parameters
    ----------
    parent_dir :
    run_base_name :
    run_ids :
    default_id :
    agent_type :
    palette :

    Returns
    -------

    """

    # TODO - plot base canvas
    default_id = run_ids[default_id]
    config = utils.config.get_config(parent_dir, run_base_name, default_id)
    canvas = draw_background(parent_dir, run_base_name, default_id)
    cmap = plt.get_cmap(palette)

    # Init plotting
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(canvas, cmap=plt.get_cmap("gray"), extent=(0, 1, 0, 1), origin="lower")

    for i, run_id in enumerate(run_ids):
        config = utils.config.get_config(parent_dir, run_base_name, run_id)
        spawn_area = np.array(config.get("environment").get(f"{agent_type}_spawn_area"))
        fixed_spawn = spawn_area[0,0] == spawn_area[1,0] and \
            spawn_area[0,1] == spawn_area[1,1]

        # TODO - Use config to plot spawn box
        if not fixed_spawn:
            rect = mpatch.Rectangle(spawn_area[0],
                                    width=spawn_area[1,0]-spawn_area[0,0],
                                    height=spawn_area[1,1]-spawn_area[0,1],
                                    ec=cmap(i), fc=cmap(i, 0.1))
            ax.add_patch(rect)
            ax.annotate(f"{run_base_name}_{run_id}",
                        (1.0, 0.0), xycoords=rect, ha="left", va="bottom",
                        color=cmap(i), fontsize=12)
        elif fixed_spawn:
            dot = mpatch.Circle(spawn_area[0], radius=0.01, fc=cmap(i, 0.8), ec="black")
            ax.add_patch(dot)
            ax.annotate(f"{run_base_name}_{run_id}", spawn_area[0],
                        (5, 5), textcoords="offset points", color=cmap(i), fontsize=12,
                        arrowprops=dict(color=cmap(i), arrowstyle="simple"))

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
        palette="Set1"
):

    base_dir = "logs"
    rep_base_name = "DQN"
    file_base_name = "eval_result.csv"

    result = read_run_level(
        base_dir, parent_dir, run_base_name, run_ids, rep_base_name, file_base_name,
        index_col=0, max_reps=max_reps
    )

    # Grouped summary
    result = result.groupby(level=["run", "rep"]).mean()

    if metric is None:
        metric = result.columns[~result.columns.str.match(r".*[xy]$")]
    elif isinstance(metric, str):
        metric = [metric]

    result.reset_index(level="run", inplace=True)
    fig = plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap("Set1")

    for i, scalar in enumerate(metric, start=1):
        ax = plt.subplot(2, 2, i)

        sns.boxplot(x="run", y=scalar, data=result, ax=ax, hue="run", palette=palette)
        sns.stripplot(x="run", y=scalar, data=result, ax=ax, alpha=0.8, jitter=True, hue="run", palette=palette)
        ax.set_title(scalar.replace("_", " ").title())
        ax.set_ylabel("")
        ax.legend().set_visible(False)

        # result.reset_index(level="run").plot.box(column=scalar, by="run", ax=ax)
        # ax.scatter(result.index.get_level_values("run").astype("str"), result[scalar], alpha=0.5)

    fig.suptitle(f"Evaluation Result\n{run_base_name} | {run_ids}")

    if save:
        plotpath = os.path.join("plot", parent_dir, f"{run_base_name}_{'-'.join([str(id) for id in run_ids])}_eval-result")
        fig.savefig(plotpath)

    if show:
        plt.show()

    return result


if __name__ == "__main__":
    # Metadata
    parent_dir = "test2"
    run_base_name = "TestWorked"
    run_ids = [1,3,4,5,8]
    max_reps = 10
    metric = None
    save = False
    show = True

    plot_info_background(
        parent_dir,
        run_base_name,
        run_ids
    )

    result = plot_eval_result(
        parent_dir,
        run_base_name,
        run_ids,
        max_reps=max_reps,
        show=show,
        save=save
    )

    # print(result.groupby(by='run').count())
