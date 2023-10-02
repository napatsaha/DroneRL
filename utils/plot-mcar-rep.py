# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:51:22 2023

@author: napat
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def q1(a):
    return np.quantile(a, q=0.25)

def q3(a):
    return np.quantile(a, q=0.75)

run_ids = [8,9,10]
labels = ["(x,vel)", "(x,vel,y)", "(x,y)"]

# Y-axis
scalars = ["ep_len_mean","ep_rew_mean"]
# scalars = ["loss"]

# X-Axis
time_units = "episodes"

parent_dir = "test1"
run_base_name = "MountainCar"
run_names = [f"{run_base_name}_{run_id}" for run_id in run_ids]

bin_width = 2000

run_df_list = []
for run in run_names:
    df_list = []
    for dirpath, dirnames, filenames in os.walk(os.path.join("../test_codes/logs", parent_dir, run)):
        if len(dirnames) > 0:
            continue

        csv_file = next(filter(lambda x: x.endswith("csv"), filenames))
        df = pd.read_csv(os.path.join(dirpath, csv_file))
        
        df_list.append(df)

    run_df = pd.concat(df_list, keys=[*range(1, len(df_list) + 1)],
                       names=["run_ids", "row_id"])
    run_df.rename(columns = lambda x: x.split("/")[1], inplace=True)
    
    run_df = run_df.loc[:, [time_units, *scalars]]
    
    # Only bin if bin width if much smaller than time units (for timesteps)
    if np.median(run_df[time_units]) > bin_width:
        run_df[time_units] = run_df[time_units].apply(lambda x: x // bin_width * bin_width)
    agg_df = run_df.groupby(by=time_units).agg(("min", q1, "median", q3, "max"))
    
    # run_df = run_df.loc[:, ["ep_len_mean", "ep_rew_mean"]]
    # agg_df = run_df.groupby(level="row_id").agg(("min", q1, "median", q3, "max"))

    run_df_list.append(agg_df)

combined_df = pd.concat(run_df_list, keys=run_names)
print(combined_df)

# Plot
cmap = mpl.colormaps["Set1"]
colors = cmap.colors[:len(run_ids)]
for scalar in scalars:
    fig, ax = plt.subplots(figsize=(15,10))
    for run, c, lab in zip(run_names, colors, labels):
        z = combined_df.loc[run, scalar]
        ax.fill_between(z.index, z['min'], z['max'], color=mpl.colors.to_rgba(c, 0.05))
        ax.fill_between(z.index, z['q1'], z['q3'], color=mpl.colors.to_rgba(c, 0.1))
        ax.plot(z.index, z['median'], color=c, label=lab)
    plt.legend(loc="upper left", title="Observations:")
    plt.xlabel(z.index.name.replace("_"," ").title())
    plt.title(f"""
              {run_base_name}
              {scalar.replace('_',' ').title()}
              Bin Width: {bin_width}
              """)
    plt.savefig(os.path.join("plot", run_base_name, f"{run_base_name}_{'_'.join(map(str, run_ids))} {scalar}.png"))
    plt.show()
