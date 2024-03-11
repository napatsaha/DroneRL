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

from envs.display import Predator

# Metadata
parent_dir = "test2"
run_base_name = "TestQvalues"
run_id = 1
rep_name = "DQN_1"
agent_name = 'predator1'
save = True
show = False

if show:
    matplotlib.use("TkAgg")

## Prepare data

# Load entire csv file
test_file = f"logs/{parent_dir}/{run_base_name}_{run_id}/{rep_name}/{agent_name}/trajectory.csv"
timestep = [20000, 40000, 60000, 80000]

data = pd.read_csv(test_file)

# Slice data frame according to episodes that envelopes the requested timesteps
# eps_match = data.episode[data.index == timestep].values[0]
eps_match = data.episode[data.index.isin(timestep)].to_list()
data = data[data.episode.isin(eps_match)]

# Number of step values for each episode (to be used in flexible axis, and episode id indexing)
# num_steps = data.shape[0]
num_steps = data.groupby("episode")["timestep"].max()

# Set index to (episode, timestep) for easy indexing
data.set_index(["episode", "timestep"], inplace=True)
idx = pd.IndexSlice # MultiIndex slicing helper

## Configurations parameters

# Get environment config
config = utils.config.get_config(parent_dir, run_base_name, run_id)
env = env_utils.create_env(config)
env.reset()

# Various cosmetic parameters
num_actions = env.action_space[0].n
CIRCLE_OFFSET = 0.05
RADIUS = 0.01
DECIMAL_PLACE = 2
TEXT_OFFSET = 0.01

# Color maps for displaying "chosen" action
activated_color = mcl.to_rgba('red')
unactivated_color = mcl.to_rgba('white')

# Other params
BASE_TITLE = "Q-Values Walking Animation"
FORMAT = "mp4"

## Trajectory

pos_cmap = plt.get_cmap("pink")
pos_norm = mcl.Normalize(vmin=0, vmax=300)

## Qvalues

qa = data[data.columns[data.columns.str.match("q.*")]].values
q_cmap = plt.get_cmap("cool")
q_normaliser = mcl.Normalize(vmin=-1, vmax=1)

# Draw Background

canvas = env.draw_canvas(draw_agents=False, return_canvas=True)
canvas.draw(env.agents[0]) # Draw prey
canvas = canvas.canvas.T

#######
# Helper function for animating

def generate_trajectory(ep, ts):
    # pos = data.iloc[:ts,:].loc[:, ["s2", "s3"]].values
    pos = data.loc[idx[ep, :ts], ["s2","s3"]].values
    pos = np.expand_dims(pos, 1)
    seg = np.concatenate([pos[:-1], pos[1:]], axis=1)

    # trj_c = data.loc[:, 'timestep'].iloc[:ts]
    trj_c = data.loc[idx[ep, :ts], :].index.get_level_values("timestep")
    trj_c = pos_norm(trj_c)
    trj_c = pos_cmap(trj_c)

    return seg, trj_c




def generate_circles(ep, ts):
    circle_patches = []
    circle_colors = []
    chosen_action = data.loc[idx[ep, ts], 'action']
    for a in range(num_actions):
        # Compute positions
        center = data.loc[idx[ep, ts], ["s2", "s3"]].values
        offset = Predator.convert_action(a)
        center = center + np.array(offset) * CIRCLE_OFFSET

        # get values and colour
        qval = data.loc[idx[ep, ts], f"q{a}"]
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

def update_lines(ep, ts):
    """
    Update lines with qvaluse up to current (episode, timestep).
    For the Secondary plot.
    """
    for act_id, line in enumerate(lines):
        # xt = data["timestep"][:frame].values
        xt = data.loc[idx[ep, :ts], :].index.get_level_values("timestep").values
        # yt = qa[:frame, act_id]
        yt = data.loc[idx[ep, :ts], f"q{act_id}"].values
        line.set_data(xt, yt)

    return lines

def reset_axis(ep, ts):
    """
    Reset axis and figure when a new episode is reached.
    Currently only changes x-axis limit for line plot, and change episode and timestep number in figure title.
    """

    global_ts = timestep[num_steps.index.to_list().index(ep)]
    fig.suptitle(f"{BASE_TITLE}\nEpisode: #{ep} | Around Timestep: #{global_ts}")

    # Secondary plot
    ax2.set_xlim([0, num_steps[ep]])


# Main animation update
def update(frame):
    """
    Animation update function, to be passed into FuncAnimation().

    """
    # global actioncols, linecols
    ep, ts = frame

    if ts == 0:
        reset_axis(ep, ts)

    # print(frame)
    circle_patches, circle_colors, circle_edges = generate_circles(ep, ts)
    actioncols.set_paths(circle_patches)
    actioncols.set_facecolor(circle_colors)
    actioncols.set_edgecolor(circle_edges)

    segments, segment_colors = generate_trajectory(ep, ts)
    linecols.set_segments(segments)
    linecols.set_color(segment_colors)

    lines = update_lines(ep, ts)

    return (linecols, actioncols, *qtexts, *lines)


##########
# Plotting
##########


# Figure init
fig = plt.figure(figsize=(15,7), tight_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[3, 4])

# First figure -- Walking animation
ax = fig.add_subplot(gs[0, 1])

ax.imshow(canvas, cmap=plt.get_cmap("gray"), extent=(0,1,0,1), origin="lower")

## INTERACTIVE
# frame = 30

circle_patches = [] #generate_circles(frame)
segments, segment_colors = [], [] #generate_trajectory(frame)

actioncols = mcol.PatchCollection(circle_patches, cmap=q_cmap)
linecols = mcol.LineCollection(segments, array=segment_colors)

ax.add_collection(linecols)
ax.add_collection(actioncols)

qtexts = [ax.text(0, 0, "", horizontalalignment="center", fontsize='x-small') for _ in range(num_actions)]
# qtexts = [ax.annotate(text="", xy=(1,1), xycoords='data',
#                       xytext=(1,1), textcoords='offset points') for _ in range(num_actions)]

# Axes formatting
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Walking animation")

# Colorbar
actioncols.set_clim(q_normaliser.vmin, q_normaliser.vmax)

cb = plt.colorbar(actioncols, fraction=0.04, pad=0.05, ax=ax)
cb.set_label("Q-Value")

## INTERACTIVE
# update(frame)

## Secondary figure -- Line plot of qvalues over timesteps
ax2 = fig.add_subplot(gs[0, 0])

lines = ax2.plot(qa)
ax2.set_ylim([q_normaliser.vmin, q_normaliser.vmax])
ax2.set_xlim([0,1])
ax2.set_xlabel("Timestep")
ax2.set_ylabel("Q-Values")
ax2.set_title("Action Q-values for states visited")

# fig.suptitle(f"{BASE_TITLE}\nEpisode: #{eps_match} | Around Timestep: #{timestep}")


## Generate Animation
frames_iterator = iter(data.index)
ani = anim.FuncAnimation(fig, func=update, frames=frames_iterator, interval=10, blit=False, save_count=len(data.index))

## Saving and Showing
if show:
    plt.show()

if save:
    f_name = f"walker_{rep_name}_{agent_name}_ts-{'-'.join([str(t) for t in timestep])}.{FORMAT.lower()}"
    ani.save(filename=f"plot/{parent_dir}/{run_base_name}_{run_id}/anim/{f_name}", writer="ffmpeg")
