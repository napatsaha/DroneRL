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

def extract_labels(changing_var, run_names):
    variable = extract_config(parent_dir=parent_dir, run_name=run_names[0],
                              name="environment")
    variable = next(filter(lambda x: x.startswith(changing_var), variable.keys()))
    labels = [extract_config(parent_dir=parent_dir, run_name=run,
                              name=["environment", variable]) for run in run_names]
    return labels


def zip_together(listlist):
    """
    Zip together list of lists into element-by-element tuples.
    e.g.
    ```
    x = [[1, 2, 3], [10, 20, 30]]
    zip_together(x)
    -> [(1, 10), (2, 20), (3, 30)]
    ```
    """
    return [(args) for args in zip(*listlist)]


parent_dir = "multi1"
run_base_name = "DoublePredator"
run_ids = [15,5,14,11,6,7,10,9,8]
changing_var = ["observation_distance_strategy","reward_distance_strategy"]
agent_names = ["prey1", "predator1"]
save = True
plot_name = "obs-rew_dist_strategies"

# Y-axis
scalars = ["loss", "ep_rew_mean"]

run_names = [f"{run_base_name}_{run_id}" for run_id in run_ids]

if plot_name is None:
    plot_name = f"{run_base_name}_{'-'.join(map(str, run_ids))}"

# Extra labels for plotting
if isinstance(changing_var, list):
    listofconfig = [extract_labels(var, run_names) for var in changing_var]
    config_pairs = zip_together(listofconfig)
    variable = tuple(changing_var)
    labels = config_pairs
else:
    variable = extract_config(parent_dir=parent_dir, run_name=run_names[0],
                              name="environment")
    variable = next(filter(lambda x: x.startswith(changing_var), variable.keys()))
    labels = [extract_config(parent_dir=parent_dir, run_name=run,
                              name=["environment", variable]) for run in run_names]

    # Sort again based on ascending values of variable
    run_ids = np.array(run_ids)[np.argsort(labels)]
    run_names = [f"{run_base_name}_{run_id}" for run_id in run_ids]
    labels = sorted(labels)


# scalars = ["loss"]

# X-Axis
time_units = "total_timesteps"
bin_width = 2000

df_dict = {name: [] for name in agent_names}

for run_name in run_names:

    df_run_dict = {name: [] for name in agent_names}

    for dirpath, dirnames, filenames in os.walk(os.path.join("logs", parent_dir, run_name)):
        current_dir = os.path.split(dirpath)[-1]

        if current_dir in agent_names:
            csv_file = next(filter(lambda x: x.endswith("csv"), filenames))
            df = pd.read_csv(os.path.join(dirpath, csv_file))
            df_run_dict[current_dir].append(df)
        else:
            continue

    for agent in agent_names:
        run_df = pd.concat(df_run_dict[agent], keys=[*range(1, len(df_run_dict[agent]) + 1)],
                           names=["run_ids", "row_id"])
        run_df.rename(columns=lambda x: x.split("/")[1], inplace=True)

        run_df = run_df.loc[:, [time_units, *scalars]]

        # Only bin if bin width if much smaller than time units (for timesteps)
        if time_units == "total_timesteps":
            run_df[time_units] = run_df[time_units].apply(lambda x: x // bin_width * bin_width)

        run_df = run_df.groupby(by=time_units).agg(("min", q1, "median", q3, "max"))

        # run_df = run_df.loc[:, ["ep_len_mean", "ep_rew_mean"]]
        # agg_df = run_df.groupby(level="row_id").agg(("min", q1, "median", q3, "max"))

        df_dict[agent].append(run_df)

for agent in agent_names:
    combined_df = pd.concat(df_dict[agent], keys=run_names)
    df_dict[agent] = combined_df

# Plot
# cmap = mpl.colormaps["viridis"]
# colors = cmap.colors[:len(run_names)]
cmap = mpl.colormaps.get_cmap("tab10")
colors = cmap(np.linspace(0,1,num=len(run_names)))
# colors = [col for i, col in enumerate(cmap.colors) if (i+1) % 4 != 0][:9]

for agent in agent_names:
    for scalar in scalars:
        fig, ax = plt.subplots(figsize=(15, 10))
        for run, c, lab in zip(run_names, colors, labels):
            z = df_dict[agent].loc[run, scalar]
            ax.fill_between(z.index, z['min'], z['max'], color=mpl.colors.to_rgba(c, 0.05))
            ax.fill_between(z.index, z['q1'], z['q3'], color=mpl.colors.to_rgba(c, 0.1))
            ax.plot(z.index, z['median'], color=c, label=lab)
        plt.legend(loc="upper left", title=variable)
        plt.xlabel(z.index.name.replace("_", " ").title())
        plt.ylabel(scalar.replace("_", " ").title())
        plt.title(f"""
        {scalar.replace("_"," ").title()}
        {run_base_name}   Agent: {agent.capitalize()}
                  """)
        if save:
            if not os.path.exists(os.path.join("plot", parent_dir)):
                os.mkdir(os.path.join("plot", parent_dir))

            file_name = f"{plot_name}_{agent}_{scalar.replace('_','-')}"
            plt.savefig(os.path.join("plot", parent_dir, file_name))
        plt.show()
