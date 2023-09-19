# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:17:06 2023

@author: napat

Download Tensorboard-generated TF-Events into readable data
"""
import os, glob, re
# from tensorflow.python.summary import 
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# from tensorboard.backend.event_processing.event_file_loader import 

def tfevent_to_csv(tf_file, csv_file):
    
    def extract_scalar(tag):
        scalar_events = ea.Scalars(tag)
        scalar_vals = [*map(lambda event: event.value, scalar_events)]
        return scalar_vals
    
    ea = EventAccumulator(tf_file)
    ea.Reload()
    labels = ea.Tags()['scalars']

    scalars = [*map(extract_scalar, labels)]
    df = pd.DataFrame(zip(*scalars), columns=labels)

    df.to_csv(csv_file, index=False)
    
    return df

def accumulate_csv(run_name, return_df=False):
    file_loader = os.walk(os.path.join("logs", run_name))
    
    if return_df: df_list = []
    
    for dirpath, dirnames, filenames in file_loader:
    
        # dirpath, dirnames, filenames = next(file_loader)
        if len(dirnames) > 0:
            continue
        run_id = re.search("_(\d+)$", dirpath).group(1)
        tf_file = next(filter(lambda x: x.startswith("events"), filenames))
        tf_file = os.path.join(dirpath, tf_file)
        csv_file = os.path.join(dirpath, f"{run_name}_{run_id.zfill(2)}.csv")
        
        df = tfevent_to_csv(tf_file, csv_file)
        if return_df:
            df_list.append(df)
            
    if return_df:
        return df_list

run_name = "DQN01_7"
run_desc = "Exploration During First 10%"
scalar_name = "rollout/ep_rew_mean"

file_loader = os.walk(os.path.join("logs", run_name))

df_list = accumulate_csv(run_name, return_df=True)
# for dirpath, dirnames, filenames in file_loader:
#     tf_file = os.path.join(dirpath, filenames[0])

scalar_df = pd.concat(map(lambda df: df[scalar_name], df_list), axis=1, ignore_index=True)
# scalar_df.isna().sum(axis=0)

# series_list = []
# for df in df_list:
#     sr = df[scalar_name]
#     series_list.append(sr)
# scalar_df = pd.concat(series_list, axis=1)
# [len(sr) for sr in series_list]

def q1(a):
    return np.quantile(a, q=0.25)

def q3(a):
    return np.quantile(a, q=0.75)

# z = scalar_df.apply(['max', 'min', 'median', q1, q3], axis=1)
z = scalar_df.apply(['max', 'min', 'median'], axis=1)

func_list = [np.max, np.min, np.median, q1, q3]
z = pd.concat([scalar_df.apply(func, axis=1) for func in func_list], axis=1)
z.columns = [func.__name__ for func in func_list]

if not os.path.exists(os.path.join("plot", run_name)):
    os.mkdir(os.path.join("plot", run_name))

fig, ax = plt.subplots(figsize=(15,10))
ax.fill_between(z.index, z['min'], z['max'], color=mpl.colors.to_rgba('gray', 0.1))
ax.fill_between(z.index, z['q1'], z['q3'], color=mpl.colors.to_rgba('gray', 0.3))
ax.plot(z.index, z['median'], color='black')
plt.title(f"{run_name}\n{scalar_name}\n{run_desc}")
plt.savefig(os.path.join("plot", run_name, f"{scalar_name.replace('/','_')}.png"))
plt.show()