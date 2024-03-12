import enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

import utils
from utils import env as env_utils

class ColorCategory(enum.Enum):
    by_steps = "Steps"
    by_episode = "Episodes"

# Metadata
parent_dir = "test2"
run_base_name = "TestQvalues"
run_id = 1
rep_name = "DQN_4"
save = True
color_by = ColorCategory.by_episode

# Fixed variables
folder = "logs"
base_file_name = "trajectory.csv"
run_name = f"{run_base_name}_{run_id}"

# Init plot path
plot_path = None
if save:
    plot_path = os.path.join("plot", parent_dir)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plot_path = os.path.join("plot", parent_dir, run_name)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

# Data
full_file_path = os.path.join(folder, parent_dir,
                              run_name, rep_name, base_file_name)
record_data = np.genfromtxt(full_file_path,
                            delimiter=',',
                            # max_rows=10000,
                            )

# record_data = record_data[record_data[:, 1] <= 10]

# trajectory = record_data[:, 4:]

# Environment -- Draw canvas
config = utils.config.get_config(parent_dir, run_base_name, run_id)
env = env_utils.create_env(config)

canvas = env.draw_canvas(draw_agents=False, return_canvas=True)
canvas = canvas.canvas.T

# Manipulate data
df = pd.DataFrame(record_data)
# Create segments
seg_ends = df.groupby(by=1).shift(-1).loc[:, [4,5]]
df2 = pd.concat([df, seg_ends], axis=1, ignore_index=True)
df2 = df2.dropna()
# Count index per episode for colors
df2[8] = df2.groupby(by=1).cumcount()
# Get num episode
num_eps = int(df2[1].max())

# Prepare data
# colors
if color_by == ColorCategory.by_steps:
    z = df2[8].values
    cb_max = env.trunc_limit
    cb_cmap = "Reds"
elif color_by == ColorCategory.by_episode:
    z = df2[1].values
    cb_max = num_eps
    cb_cmap = "Greens"
# x, y segments
segments = np.concatenate([df2.iloc[:, 4:6], df2.iloc[:, 6:8]], axis=1)
segments = segments.reshape(-1, 2, 2)
# to_exclude = df.reset_index().groupby(by=1).last()['index']




# Plotting

lines = LineCollection(segments, array=z, cmap=plt.get_cmap(cb_cmap), alpha=0.1)

fig, ax = plt.subplots(figsize=(12,12))
# ax = plt.gca()

# Draw main components
ax.imshow(canvas, cmap=plt.get_cmap("gray"))
ax.add_collection(lines)
cb = plt.colorbar(lines)

# Axis cosmetics
ax.set_xlim(0, env.resolution)
ax.set_ylim(0, env.resolution)
ax.set_xticks([])
ax.set_yticks([])

# Colorbar
cb.solids.set(alpha=1)
cb.ax.set_ylim(0, cb_max)
cb.set_label(color_by.value)

ax.set_title(f"{env.agent_list[0].title()} Composite Trajectory by {color_by.value}" + "\n" +
             f"{run_name} | {rep_name} | {num_eps} Episodes")

if save:
    plt.savefig(os.path.join(plot_path, f"{rep_name}-composite-trajectory-{color_by.value.lower()}.png"))

plt.show()