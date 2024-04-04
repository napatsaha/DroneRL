import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.tools import read_run_level


def plot_eval_result(
        parent_dir,
        run_base_name,
        run_ids,
        *,
        metric=None,
        save=False,
        show=True,
        max_reps=None,

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

    for i, scalar in enumerate(metric, start=1):
        ax = plt.subplot(2, 2, i)

        sns.boxplot(x="run", y=scalar, data=result, ax=ax, color="white")
        sns.stripplot(x="run", y=scalar, data=result, ax=ax, alpha=0.8, jitter=True, color="grey")
        ax.set_title(scalar.replace("_", " ").title())
        ax.set_ylabel("")

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
    run_ids = [1, 3, 4, 5, 8]
    max_reps = 10
    metric = None
    save = True
    show = True

    result = plot_eval_result(
        parent_dir,
        run_base_name,
        run_ids,
        max_reps=max_reps,
        show=show,
        save=save
    )

    print(result.groupby(by='run').count())
