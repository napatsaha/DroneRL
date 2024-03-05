import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

# from envs import environment_v1
# from utils import get_config
import utils
from utils import env as env_utils

# Metadata
parent_dir = "test2"
run_base_name = "TestLog"
run_id = 3
rep_name = "DQN_1"
save = True

# Fixed variables
folder = "logs"
base_file_name = "trajectory.csv"
run_name = f"{run_base_name}_{run_id}"

# Init path
plot_path = None
if save:
    plot_path = os.path.join("plot", parent_dir)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plot_path = os.path.join("plot", parent_dir, run_name)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)


full_file_path = os.path.join(folder, parent_dir,
                              run_name, rep_name, base_file_name)

# Data
record_data = np.genfromtxt(full_file_path,
                            delimiter=',',
                            # max_rows=10000,
                            )

# record_data = record_data[record_data[:, 1] <= 10]

# trajectory = record_data[:, 4:]

# Manipulate data
df = pd.DataFrame(record_data)
# Create segments
seg_ends = df.groupby(by=1).shift(-1).loc[:, [4,5]]
df2 = pd.concat([df, seg_ends], axis=1, ignore_index=True)
df2 = df2.dropna()
# Count index per episode for colors
df2[8] = df2.groupby(by=1).cumcount()

# Extract data
z = df2[8].values # colors
# x, y segments
segments = np.concatenate([df2.iloc[:, 4:6], df2.iloc[:, 6:8]], axis=1)
segments = segments.reshape(-1, 2, 2)
# to_exclude = df.reset_index().groupby(by=1).last()['index']
num_eps = int(df2[1].max())

# Environment
config = utils.config.get_config(parent_dir, run_base_name, run_id)
env = env_utils.create_env(config)

canvas = env.draw_canvas(draw_agents=False, return_canvas=True)
canvas = canvas.canvas.T

# Plotting

lines = LineCollection(segments, array=z, cmap=plt.get_cmap('Reds'), alpha=0.1)

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
cb.ax.set_ylim(0, env.trunc_limit)
cb.set_label("Step in episode")

ax.set_title(f"{env.agent_list[0].title()} Composite Trajectory" + "\n" +
             f"{run_name} | {rep_name} | {num_eps} Episodes")

if save:
    plt.savefig(os.path.join(plot_path, f"{rep_name}-composite-trajectory.png"))

plt.show()