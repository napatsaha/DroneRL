import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mcol
import matplotlib.patches as patch
import matplotlib.colors as mcl
import matplotlib.animation as anim


import utils
from utils import env as env_utils

import matplotlib
from IPython.display import HTML


from envs.display import Predator

# Metadata
parent_dir = "test2"
run_base_name = "TestQvalues"
run_id = 1
rep_name = "DQN_1"
agent_name = 'predator1'
save = False
show = True

if show:
    matplotlib.use("TkAgg")

# Prepare data

test_file = f"logs/{parent_dir}/{run_base_name}_{run_id}/{rep_name}/{agent_name}/trajectory.csv"
timestep = 80000

data = pd.read_csv(test_file)

eps_match = data.episode[data.index == timestep].values[0]

data = data[data.episode == eps_match]

num_steps = data.shape[0]

# Configurations parameters
config = utils.config.get_config(parent_dir, run_base_name, run_id)
env = env_utils.create_env(config)
env.reset()

num_actions = env.action_space[0].n
CIRCLE_OFFSET = 0.05
RADIUS = 0.01
DECIMAL_PLACE = 2
TEXT_OFFSET = 0.01

activated_color = mcl.to_rgba('red')
unactivated_color = mcl.to_rgba('white')

BASE_TITLE = "Q-Values Walking Animation"
FORMAT = "mp4"

# Trajectory

pos_cmap = plt.get_cmap("pink")
pos_norm = mcl.Normalize(vmin=0, vmax=300)

def generate_trajectory(ts):
    pos = data.iloc[:ts,:].loc[:, ["s2", "s3"]].values
    pos = np.expand_dims(pos, 1)
    seg = np.concatenate([pos[:-1], pos[1:]], axis=1)

    trj_c = data.loc[:, 'timestep'].iloc[:ts]
    trj_c = pos_norm(trj_c)
    trj_c = pos_cmap(trj_c)

    return seg, trj_c


# Qvalues

qa = data[data.columns[data.columns.str.match("q.*")]].values
q_cmap = plt.get_cmap("cool")
q_normaliser = mcl.Normalize(vmin=-1, vmax=1)

def generate_circles(ts):
    circle_patches = []
    circle_colors = []
    chosen_action = data['action'].iloc[ts]
    for a in range(num_actions):
        # Compute positions
        center = data.loc[:, ["s2", "s3"]].iloc[ts-1].values
        offset = Predator.convert_action(a)
        center = center + np.array(offset) * CIRCLE_OFFSET

        # get values and colour
        qval = data.loc[:, f"q{a}"].iloc[ts-1]
        c = q_cmap(q_normaliser(qval))

        # Create circles
        circle = patch.Circle(center.squeeze(), radius=RADIUS)

        # Write text
        qtexts[a].set(position=center + np.array([0, TEXT_OFFSET]), text=f"{qval:.{DECIMAL_PLACE}f}", color=c)
        # Append patch and color
        circle_patches.append(circle)
        circle_colors.append(c)

    circle_edge = [unactivated_color if a != chosen_action else activated_color for a in range(num_actions)]
    return circle_patches, circle_colors, circle_edge


## Animation update
def update(frame):
    # global actioncols, linecols

    # print(frame)
    circle_patches, circle_colors, circle_edges = generate_circles(frame)
    actioncols.set_paths(circle_patches)
    actioncols.set_facecolor(circle_colors)
    actioncols.set_edgecolor(circle_edges)

    segments, segment_colors = generate_trajectory(frame)
    linecols.set_segments(segments)
    linecols.set_color(segment_colors)

    return (linecols, actioncols, *qtexts)


## Plotting

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()

# Background

canvas = env.draw_canvas(draw_agents=False, return_canvas=True)
canvas.draw(env.agents[0]) # Draw prey
canvas = canvas.canvas.T
ax.imshow(canvas, cmap=plt.get_cmap("gray"), extent=(0,1,0,1), origin="lower")

## INTERACTIVE
# frame = 30

circle_patches = [] #generate_circles(frame)
segments, segment_colors = [], [] #generate_trajectory(frame)

actioncols = mcol.PatchCollection(circle_patches, cmap=q_cmap)
linecols = mcol.LineCollection(segments, array=segment_colors)

ax.add_collection(linecols)
ax.add_collection(actioncols)

qtexts = [ax.text(0,0,"", horizontalalignment="center", fontsize='x-small') for _ in range(num_actions)]
# qtexts = [ax.annotate(text="", xy=(1,1), xycoords='data',
#                       xytext=(1,1), textcoords='offset points') for _ in range(num_actions)]

# Axes formatting
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"{BASE_TITLE}\nEpisode: #{eps_match} | Around Timestep: #{timestep}")

# Colorbar
actioncols.set_clim(q_normaliser.vmin, q_normaliser.vmax)

cb = plt.colorbar(actioncols, fraction=0.04, pad=0.05)
cb.set_label("Q-Value")

## INTERACTIVE
# update(frame)

ani = anim.FuncAnimation(fig, func=update, frames=num_steps, interval=100, blit=True)

if show:
    plt.show()

if save:
    ani.save(filename=f"plot/{parent_dir}/{run_base_name}_{run_id}/anim/walker_{rep_name}_{agent_name}_ep-{eps_match}.{FORMAT.lower()}", writer="ffmpeg")
