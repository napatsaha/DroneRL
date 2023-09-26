import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Iterable

run_name = ["MountainCar_6", "MountainCar_7"]
labels = ["(x,y) positions", "x position only"]
run_dir = "test1"
scalar_names = ['rollout/ep_len_mean', 'rollout/ep_rew_mean']

# Case of plotting for single run only
if not isinstance(run_name, list) or len(run_name) < 2:
    if isinstance(run_name, list) and len(run_name) == 1:
        run_name = run_name[0]
    file = os.path.join("../test_codes/logs", run_dir, run_name, "progress.csv")

    progress = pd.read_csv(file)

    for scalar in scalar_names:
        plt.plot(progress['time/total_timesteps'], progress[scalar])
        plt.title(scalar.split("/")[-1] +"\n" + run_name)
        plt.show()
elif isinstance(run_name, list):
    if labels is None or len(labels) != len(run_name):
        labels = run_name
    for scalar in scalar_names:
        for run, tag in zip(run_name, labels):
            file = os.path.join("../test_codes/logs", run_dir, run, "progress.csv")

            progress = pd.read_csv(file)

            plt.plot(progress['time/total_timesteps'], progress[scalar], label=tag)
        plt.title("MountainCar\n" + scalar.split("/")[-1])
        plt.xlabel("Timesteps")
        plt.ylabel(scalar.split("/")[-1])
        plt.legend(title="Observation contains:")
        plt.show()

