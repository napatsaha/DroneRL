# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:51:22 2023

@author: napat
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

run_ids = [8,9]

parent_dir = "test1"
run_base_name = "MountainCar"
run_names = [f"{run_base_name}_{run_id}" for run_id in run_ids]

run_df_list = []
for run in run_names:
    df_list = []
    for dirpath, dirnames, filenames in os.walk(os.path.join("logs", parent_dir, run)):
        if len(dirnames) > 0:
            continue

        csv_file = next(filter(lambda x: x.endswith("csv"), filenames))
        df = pd.read_csv(os.path.join(dirpath, csv_file))
        print(df)
        df_list.append(df)

    run_df = pd.concat(df_list, keys=[*range(1, len(df_list) + 1)],
                       names=["run_id", "row_id"])
    run_df = run_df.loc[:, ["time/total_timesteps", "rollout/ep_len_mean", "rollout/ep_rew_mean"]]
    run_df["time/total_timesteps"] = run_df["time/total_timesteps"].apply(lambda x: x // 1000 * 1000)
    agg_df = run_df.groupby(by="time/total_timesteps").agg(("min", "median", "max"))

    run_df_list.append(agg_df)

