# -*- coding: utf-8 -*-
"""
Aggregate all repetitions for a single run for each of predator and prey and plot them against each other.
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.config import extract_config

def q1(a):
    return np.quantile(a, q=0.25)

def q3(a):
    return np.quantile(a, q=0.75)


def plot(parent_dir, run_base_name, run_ids, rep_name=None,
         agent_names=None, save=False, sort=False,
         scalars=("ep_rew_mean",), time_units="total_timesteps",
         bin_width=2000, changing_var=None,
         plot_title=None, legend_title=None, labels=None):

    ##########
    # Metadata
    ##########

    run_names = [f"{run_base_name}_{run_id}" for run_id in run_ids]
    collated_run_names = f"{run_base_name}_{'-'.join(map(str, run_ids))}"

    agent_names = ["predator"] if agent_names is None else agent_names
    # Future
    # agent_names = extract_config(parent_dir=parent_dir, run_name=run_names[0],
    #                              name="agent_names")

    # Custom labelling by variable name grouping
    if changing_var is not None:
        variable = None
        if isinstance(changing_var, str):
            variable = changing_var.title()
        elif isinstance(changing_var, (list, tuple)):
            variable = '/'.join(changing_var).title()

        if legend_title is None:
            legend_title = variable

        if plot_title is None:
            plot_title = f"Effect of changing {variable}"

        if labels is None:
            labels = [extract_config(parent_dir=parent_dir, run_name=run,
                                     name=changing_var) for run in run_names]
            # Sort again based on ascending values of variable
            if sort:
                run_ids = np.array(run_ids)[np.argsort(labels)]
                run_names = [f"{run_base_name}_{run_id}" for run_id in run_ids]
                labels = sorted(labels)
    else:
        if labels is None:
            labels = run_names
        plot_title = parent_dir
        legend_title = run_base_name

    # variable = changing_var


    ##############
    # Collate data
    ##############

    df_dict = {name: [] for name in agent_names}

    for run_name in run_names:

        df_run_dict = {name: [] for name in agent_names}

        # Collect csv files from matching "agent_names" and "rep_names"
        for dirpath, dirnames, filenames in os.walk(os.path.join("logs", parent_dir, run_name)):
            current_dir = os.path.split(dirpath)[-1]
            above_current_dir = str.split(dirpath, os.path.sep)[-2]

            if current_dir in agent_names:
                if rep_name is None or above_current_dir in rep_name:
                    # Read file and append to dictionary
                    # csv_file = next(filter(lambda x: x.endswith("csv"), filenames))
                    csv_file = "progress.csv"
                    df = pd.read_csv(os.path.join(dirpath, csv_file))
                    df_run_dict[current_dir].append(df)
                else:
                    continue
            else:
                continue

        # Collate data from same "agent_names" across "rep_names"
        for agent in agent_names:
            run_df = pd.concat(df_run_dict[agent], keys=[*range(1, len(df_run_dict[agent]) + 1)],
                               names=["run_ids", "row_id"])
            run_df.rename(columns=lambda x: x.split("/")[1], inplace=True)

            run_df = run_df.loc[:, [time_units, *scalars]]

            # Bin x-axis across reps/samples according to bin-width
            # Only bin if bin width if much smaller than time units (for timesteps)
            if time_units == "total_timesteps":
                run_df[time_units] = run_df[time_units].apply(lambda x: x // bin_width * bin_width)

            run_df = run_df.groupby(by=time_units).agg(("min", q1, "median", q3, "max"))

            # run_df = run_df.loc[:, ["ep_len_mean", "ep_rew_mean"]]
            # agg_df = run_df.groupby(level="row_id").agg(("min", q1, "median", q3, "max"))

            df_dict[agent].append(run_df)

    # Collate across "run_ids" for same "agent_names"
    for agent in agent_names:
        combined_df = pd.concat(df_dict[agent], keys=run_names)
        df_dict[agent] = combined_df


    ##########
    # Plotting
    ##########

    # cmap = mpl.colormaps["viridis"]
    cmap = mpl.colormaps.get_cmap("Set2")
    # colors = cmap(np.linspace(0,1,num=len(run_names)))
    colors = cmap.colors[:len(run_names)]

    # Separate plots for each "agent_names"
    for agent in agent_names:
        for scalar in scalars:
            fig, ax = plt.subplots(figsize=(15, 10))
            for run, c, lab in zip(run_names, colors, labels):
                z = df_dict[agent].loc[run, scalar]
                ax.fill_between(z.index, z['min'], z['max'], color=mpl.colors.to_rgba(c, 0.05))
                ax.fill_between(z.index, z['q1'], z['q3'], color=mpl.colors.to_rgba(c, 0.1))
                ax.plot(z.index, z['median'], color=c, label=lab)
            plt.legend(loc="upper left", title=legend_title)
            plt.xlabel(z.index.name.replace("_", " ").title())
            plt.ylabel(scalar.replace("_", " ").title())
            plt.title(f"""
            {plot_title}
            {scalar.replace("_"," ").title()} | {collated_run_names} | {agent.capitalize()}
                      """.replace("\t",""))
            # Save figure
            if save:
                if not os.path.exists(os.path.join("plot", parent_dir)):
                    os.mkdir(os.path.join("plot", parent_dir))

                file_name = f"{collated_run_names}_{scalar.replace('_','-')}_{agent}"
                plt.savefig(os.path.join("plot", parent_dir, file_name))

            plt.show()


if __name__ == "__main__":
    parent_dir = "test2"
    run_base_name = "TestQvalues"
    run_ids = [1]
    rep_name = ["DQN_1", "DQN_3", "DQN_4"]
    # changing_var = None
    sort = False
    save = True

    plot_title = None # "Epsilon-greedy vs variations of Softmax probabilistic Exploration"
    legend_title = None #"Reward based on:" # "Exploration Strategy:"
    changing_var = None# ("agent", "probabilistic")
    labels = None #("Distance only", "Ray Trigger", "Both") #("Epsilon-greedy", "Softmax-Greedy", "Epsilon-Softmax", "Plain Softmax")

    # Y-axis
    scalars = ["ep_rew_mean", "ep_len_mean", "loss", "exploration_rate"]
    # scalars = ["loss"]

    # X-Axis
    time_units = "total_timesteps"
    bin_width = 2000

    plot(parent_dir, run_base_name, run_ids, rep_name,
         agent_names=["predator1"],
         scalars=scalars, changing_var=changing_var,
         plot_title=plot_title, legend_title=legend_title, labels=labels)
