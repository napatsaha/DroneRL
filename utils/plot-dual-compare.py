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


parent_dir = "dual1"
run_base_name = "DualDrone"
run_ids = [8, 7, 9]
changing_var = "predator"
agent_names = ["predator","prey"]
save = True

run_names = [f"{run_base_name}_{run_id}" for run_id in run_ids]

# Extra labels for plotting
variable = extract_config(parent_dir=parent_dir, run_name=run_names[0],
                          name="environment")
variable = next(filter(lambda x: x.startswith(changing_var), variable.keys()))
labels = [extract_config(parent_dir=parent_dir, run_name=run,
                          name=["environment", variable]) for run in run_names]
# Y-axis
scalars = ["ep_len_mean", "ep_rew_mean", "loss"]
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
cmap = mpl.colormaps["Set1"]
colors = cmap.colors[:len(run_names)]
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
        plt.title(f"""
                  {agent}
                  {scalar.replace('_', ' ').title()}
                  """)
        if save:
            if not os.path.exists(os.path.join("plot", parent_dir)):
                os.mkdir(os.path.join("plot", parent_dir))

            plt.savefig(os.path.join("plot", parent_dir, f"{run_name}_{'-'.join(map(str, run_ids))}_{agent}_{scalar.replace('_','-')}"))
        plt.show()
