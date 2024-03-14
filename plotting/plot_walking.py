from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mcol
import matplotlib.patches as patch
import matplotlib.colors as mcl
import matplotlib.animation as anim
from tqdm.autonotebook import tqdm

import utils
from utils import env as env_utils
from utils.plot_utils import ACTION_DICT_CARDINAL

import matplotlib

from envs.display import Predator


def plot_walking(
        parent_dir: str,
        run_base_name: str,
        run_id: int,
        rep_name: str,
        agent_name: str,
        timestep: int | Iterable[int],
        *,
        save: bool = False,
        show: bool = True,
        position_columns: Tuple[str] = ("s2", "s3"),
        circle_offset: float = 0.05,
        radius: float = 0.01,
        decimal_place: int = 2,
        text_offset: float = 0.01,
        qval_cmap: str = "cool",
        trajectory_cmap: str = "pink",
        action_cmap: str = "Set1",
        chosen_action_color: str = "red",
        non_chosen_action_color: str = "white",
        base_title: Optional[str] = None,
        output_format: str = "mp4",
        output_writer: str = "ffmpeg"
) -> None:
    """

    Parameters
    ----------
    parent_dir : str
        Higher directory name of run (e.g. test1, colli1)
    run_base_name :
        Base Run Name (e.g. TestLog)
    run_id :
        Specific Run ID (e.g. 4)
    rep_name :
        Name for this Repetition / Trial (e.g. DQN_1)
    agent_name :
        Agent nameID (e.g. predator1)
    timestep : int | Iterable[int]
        Specify which timestep(s) to plot animation. Can be one of two things:
        - If an integer n, will assume n equally-spaced timesteps throughout the length of the training session \
        (including beginning and end)
        - If a list or tuple, will find the corresponding episodes that include those timesteps
    save : bool
        Whether to save the animation as a file
    show : bool
        Whether to visualise immediately
    position_columns :
        Column names where the position of the agent is located (in the trajectory file)
    circle_offset :
        Distance the action circles are spaced apart from each other (in width of canvas [0-1])
    radius :
        Size of the action circles
    decimal_place :
        Decimal places for the text display of q-values above the action circles.
    text_offset :
        How far the text should be placed above the action circles.
    qval_cmap :
        Colour map for the q-values action circles
    trajectory_cmap :
        Colour map for the trailing trajectory of the agent's path
    action_cmap :
        Colour map for the different actions on the line plot (discrete)
    chosen_action_color :
        Colour of the edge of the action circle for the chosen action at that timestep
    non_chosen_action_color :
        Edge colour for all actions not chosen at that timestep
    base_title :
        Title for the entire figure. Defaults to "Q-Values Walking Animation" if None.
    output_format :
        File format for output file if save=True
    output_writer : str | MovieWriter
        File writer accepted by matplotlib.animation.FuncAnimation.save(). Refer here[!https://matplotlib.org/stable/users/explain/animations/animations.html#animation-writers]
        for supported formats and writers.

    Returns
    -------
    None

    """

    # Metadata
    # parent_dir = "test2"
    # run_base_name = "TestQvalues"
    # run_id = 1
    # rep_name = "DQN_4"
    # agent_name = 'predator1'
    # save = True
    # show = False
    # timestep = 5 # Either integer (to create equally spaced timesteps), or list of timesteps

    ## Configurations parameters
    CIRCLE_OFFSET = circle_offset
    RADIUS = radius
    DECIMAL_PLACE = decimal_place
    TEXT_OFFSET = text_offset

    QVAL_CMAP = qval_cmap
    TRAJECT_CMAP = trajectory_cmap
    ACTION_CMAP = action_cmap

    # Color maps for displaying "chosen" action
    activated_color = mcl.to_rgba(chosen_action_color)
    unactivated_color = mcl.to_rgba(non_chosen_action_color)

    # Other params
    BASE_TITLE = "Q-Values Walking Animation" if base_title is None else base_title
    FORMAT = output_format

    if show:
        matplotlib.use("TkAgg")

    # Get environment config
    config = utils.config.get_config(parent_dir, run_base_name, run_id)
    env = env_utils.create_env(config)
    env.reset()

    # Various cosmetic parameters
    num_actions = env.action_space[0].n

    # Assertions
    assert config.get("agent").get("should_log_policy_trajectory") is True, \
        "Policy trajectory must be logged to plot this animation."


    ## Prepare data
    # Load entire csv file
    test_file = f"logs/{parent_dir}/{run_base_name}_{run_id}/{rep_name}/{agent_name}/trajectory.csv"
    data = pd.read_csv(test_file)

    # Process if timestep is an int
    max_timestep = data.index.max()
    if isinstance(timestep, int):
        timestep = np.linspace(0, max_timestep, num=timestep, dtype='int')

    # Slice data frame according to episodes that envelopes the requested timesteps
    # eps_match = data.episode[data.index == timestep].values[0]
    eps_match = data.episode[data.index.isin(timestep)].to_list()
    data = data[data.episode.isin(eps_match)]

    # Number of step values for each episode (to be used in flexible axis, and episode id indexing)
    # num_steps = data.shape[0]
    num_steps = data.groupby("episode")["timestep"].max()

    # Set index to (episode, timestep) for easy indexing
    data.reset_index(inplace=True)
    data.rename(columns={'index': "global_timestep"}, inplace=True)
    data.set_index(["episode", "timestep"], inplace=True)
    idx = pd.IndexSlice # MultiIndex slicing helper

    ## Trajectory

    pos_cmap = plt.get_cmap(TRAJECT_CMAP)
    pos_norm = mcl.Normalize(vmin=0, vmax=env.trunc_limit)

    ## Qvalues

    # qa = data[data.columns[data.columns.str.match("q.*")]].values
    q_cmap = plt.get_cmap(QVAL_CMAP)
    q_normaliser = mcl.Normalize(vmin=-1, vmax=env.reward_mult)

    # Draw Background

    canvas = env.draw_canvas(draw_agents=False, return_canvas=True)
    canvas.draw(env.agents[0]) # Draw prey
    canvas = canvas.canvas.T

    ########
    # Helper function for animating
    ########


    def generate_trajectory(ep, ts):
        # pos = data.iloc[:ts,:].loc[:, ["s2", "s3"]].values
        pos = data.loc[idx[ep, :ts], position_columns].values
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
            center = data.loc[idx[ep, ts], position_columns].values
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

        # global_ts = timestep[num_steps.index.to_list().index(ep)]
        ts_start, ts_end = data.loc[idx[ep,:], "global_timestep"].agg(lambda x: (x.min(), x.max()))
        fig.suptitle(f"{BASE_TITLE}\nEpisode: #{ep} | Around Timestep: #{ts_start}-{ts_end}")

        # Secondary plot
        # ax2.clear()
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

        update_lines(ep, ts)

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

    # Draw lines manually to set labels, color
    # lines = ax2.plot(qa)
    line_cmap = plt.get_cmap(ACTION_CMAP)
    lines = []
    for a in range(num_actions):
        line, = ax2.plot(np.arange(10), np.random.uniform(-1,1,10), color=line_cmap(a),
                         label=ACTION_DICT_CARDINAL[a])
        lines.append(line)

    # Set static axis properties
    ax2.set_ylim([q_normaliser.vmin, q_normaliser.vmax])
    # ax2.set_xlim([0, env.trunc_limit])
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Q-Values")
    ax2.set_title("Action Q-values for states visited")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc="lower right", title="Action")

    # fig.suptitle(f"{BASE_TITLE}\nEpisode: #{eps_match} | Around Timestep: #{timestep}")


    ## Generate Animation
    frames_iterator = iter(data.index)
    ani = anim.FuncAnimation(fig, func=update, frames=frames_iterator, interval=40, blit=False, save_count=len(data.index))

    ## Saving and Showing
    if show:
        plt.show()

    if save:
        bar = tqdm(total=len(data.index))
        f_name = f"walker_{rep_name}_{agent_name}_ts-{'-'.join([str(t) for t in timestep])}.{FORMAT.lower()}"
        ani.save(filename=f"plot/{parent_dir}/{run_base_name}_{run_id}/anim/{f_name}", writer=output_writer,
                 progress_callback=lambda i, n: bar.update())
        bar.close()


if __name__ == "__main__":
    # Metadata
    parent_dir = "test2"
    run_base_name = "TestQvalues"
    run_id = 1
    rep_name = "DQN_4"
    agent_name = 'predator1'
    save = False
    show = True
    timestep = 3

    plot_walking(
        parent_dir,
        run_base_name,
        run_id,
        rep_name,
        agent_name,
        timestep,
        save=save,
        show=show
    )