import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Iterable

runs = ["test1/MountainCar_9/DQN_1", "test2/MountainCar_4"]
labels = ["Stable-Baselines", "Custom-DQN"]
scalar_names = ['rollout/ep_len_mean', 'rollout/ep_rew_mean']
title = "Comparison between SB3 and Custom Trainers"



if labels is None or len(labels) != len(runs):
    labels = runs
for scalar in scalar_names:
    for run, tag in zip(runs, labels):
        file = os.path.join("logs", run, "progress.csv")

        progress = pd.read_csv(file)

        plt.plot(progress['time/episodes'], progress[scalar], label=tag)
    plt.title(f"{title}\n{scalar.split('/')[-1].title()}")
    plt.xlabel("Episode")
    plt.ylabel(scalar.split("/")[-1].title())
    plt.legend()
    plt.show()

