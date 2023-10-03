# -*- coding: utf-8 -*-
"""
Aggregate all repetitions for a single run for each of predator and prey and plot them against each other.
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def q1(a):
    return np.quantile(a, q=0.25)

def q3(a):
    return np.quantile(a, q=0.75)


parent_dir = "dual1"
run_base_name = "DualDrone"
run_id = 11
agent_names = ["predator", "prey"]
labels = agent_names
run_name = f"{run_base_name}_{run_id}"
save = False

# Y-axis
scalars = ["ep_len_mean", "ep_rew_mean", "loss"]
# scalars = ["loss"]

# X-Axis
time_units = "episodes"
bin_width = 2000

df_dict = {name: [] for name in agent_names}

for dirpath, dirnames, filenames in os.walk(os.path.join("logs", parent_dir, run_name)):
    current_dir = os.path.split(dirpath)[-1]

    if current_dir in agent_names:
        csv_file = next(filter(lambda x: x.endswith("csv"), filenames))
        df = pd.read_csv(os.path.join(dirpath, csv_file))
        df_dict[current_dir].append(df)
    else:
        continue

for agent in agent_names:
    run_df = pd.concat(df_dict[agent], keys=[*range(1, len(df_dict[agent]) + 1)],
                       names=["run_ids", "row_id"])
    run_df.rename(columns=lambda x: x.split("/")[1], inplace=True)

    run_df = run_df.loc[:, [time_units, *scalars]]

    # Only bin if bin width if much smaller than time units (for timesteps)
    if np.median(run_df[time_units]) > bin_width:
        run_df[time_units] = run_df[time_units].apply(lambda x: x // bin_width * bin_width)

    run_df = run_df.groupby(by=time_units).agg(("min", q1, "median", q3, "max"))

    # run_df = run_df.loc[:, ["ep_len_mean", "ep_rew_mean"]]
    # agg_df = run_df.groupby(level="row_id").agg(("min", q1, "median", q3, "max"))

    df_dict[agent] = run_df

combined_df = pd.concat(df_dict.values(), keys=agent_names)
print(combined_df)

# Plot
cmap = mpl.colormaps["Set1"]
colors = cmap.colors[:len(agent_names)]
for scalar in scalars:
    fig, ax = plt.subplots(figsize=(15, 10))
    for run, c, lab in zip(agent_names, colors, labels):
        z = combined_df.loc[run, scalar]
        ax.fill_between(z.index, z['min'], z['max'], color=mpl.colors.to_rgba(c, 0.05))
        ax.fill_between(z.index, z['q1'], z['q3'], color=mpl.colors.to_rgba(c, 0.1))
        ax.plot(z.index, z['median'], color=c, label=lab)
    plt.legend(loc="upper left", title="Observations:")
    plt.xlabel(z.index.name.replace("_", " ").title())
    plt.title(f"""
              {run_name}
              {scalar.replace('_', ' ').title()}
              Bin Width: {bin_width}
              """)
    if save:
        if not os.path.exists(os.path.join("plot", run_base_name)):
            os.mkdir(os.path.join("plot", run_base_name))

        plt.savefig(os.path.join("plot", run_base_name, f"{parent_dir}-{run_name}-{'-'.join(map(str, agent_names))}_{scalar.replace('_','-')}_by-{time_units}"))
    plt.show()
