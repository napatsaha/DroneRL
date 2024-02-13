import os

import numpy as np
import yaml
import torch as th
import matplotlib.pyplot as plt

from algorithm import ALG_DICT
from envs import ENV_DICT

parent_dir = "test1"
run_base_name = "TestAlgo"
run_id = 7
rep_name = "DQN_1"
# timestep = "050000"
# timestep = None
save = True

prey_spawn_pos = 0.5,0.5

run_name = f"{run_base_name}_{run_id}"
config_file = os.path.join("config", parent_dir, f"{run_name}.yaml")
with open(config_file, "r") as file:
    config = yaml.load(file, yaml.SafeLoader)

AgentClass = ALG_DICT[config["agent_class"]]
EnvironmentClass = ENV_DICT[config["environment_class"]]

if save:
    plot_path = os.path.join("plot", parent_dir)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plot_path = os.path.join("plot", parent_dir, run_name)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

env = EnvironmentClass(**config["environment"])
env.reset()
env.agents[0].set_position(*np.array(prey_spawn_pos)*env.canvas_width)
# Draw only prey and obstacles
env.draw_canvas()
env.canvas.clear()
for obstacle in env.obstacle_list:
    env.canvas.draw(obstacle)
env.canvas.draw(env.agents[0])
canvas = env.canvas.canvas

agent = AgentClass(env, **config["agent"])

learning_starts = config["agent"]['learning_starts']
save_interval = config['learn']['save_interval']
total_timesteps = config['learn']['total_timesteps']

timestep_list = np.arange(learning_starts+save_interval, total_timesteps+1, save_interval)

for timestep in timestep_list:
    if not isinstance(timestep, str):
        digits = len(str(total_timesteps))
        timestep = f"{timestep:0{digits}}"

    model_file = os.path.join("model", parent_dir, run_name)
    agent.load(model_file, rep_name, timestep)

    model_name = " | ".join([run_name, rep_name, timestep])

    policy = agent.agents['predator1']
    policy.q_net = policy.q_net.cpu()

    xx = th.arange(0,1,0.02)
    yy = xx

    X, Y = th.meshgrid(xx,yy, indexing='xy')
    XY = th.stack([X,Y], dim=2)

    prey_pos = th.tensor(prey_spawn_pos).expand(50,50,-1)

    with th.no_grad():
        inp = th.cat([prey_pos, XY], dim=2)
        q_values = policy.q_net(inp)

        Z_qmax = q_values.max(dim=2)[0]
        Z_qmean = q_values.mean(dim=2)
        Z_var = q_values.var(dim=2)
        Z_action = q_values.argmax(dim=2)

    fig, ax = plt.subplots(1,2, figsize=(15,5))

    ctr = ax[0].contourf(X, Y, Z_qmax, alpha=0.5)
    ax[0].imshow(canvas.T, extent=(0,1,0,1), cmap=plt.get_cmap("gray"), origin="lower")
    plt.colorbar(ctr, ax=ax[0])
    ax[0].set_title("Max Q-Value" + "\n" + model_name)
    ax[0].axis("off")
    ctr.set_clim(0,1)
    # plt.show()

    # ctr = plt.contourf(X, Y, Z_qmax, alpha=0.5)
    # plt.imshow(env.canvas.canvas.T, extent=(0,1,0,1), cmap=plt.get_cmap("gray"), origin="lower")
    # plt.colorbar(ctr)
    # plt.title("Max Q-Value" + "\n" + model_name)
    # plt.axis("off")
    # plt.show()
    #
    # plt.imshow(env.canvas.canvas.T, extent=(0,1,0,1), cmap=plt.get_cmap("gray"), origin="lower")
    # ctr = plt.contourf(X, Y, Z_qmean, alpha=0.5)
    # plt.colorbar(ctr)
    # plt.title("Mean Q-Value" + "\n" + model_name)
    # plt.axis("off")
    # plt.show()
    #
    # plt.imshow(env.canvas.canvas.T, extent=(0,1,0,1), cmap=plt.get_cmap("gray"), origin="lower")
    # ctr = plt.contourf(X, Y, Z_var, alpha=0.5)
    # plt.colorbar(ctr)
    # plt.title("Variance Q-Value" + "\n" + model_name)
    # plt.axis("off")
    # plt.show()

    action_dict = {
        0: "Stationary",
        1: "Up",
        2: "Left",
        3: "Down",
        4: "Right"
    }
    formatter = plt.FuncFormatter(lambda val, pos: action_dict[val])
    cmap = plt.get_cmap("viridis", 5)

    ax[1].imshow(canvas.T, extent=(0,1,0,1), cmap=plt.get_cmap("gray"), origin="lower")
    ctr = ax[1].contourf(X, Y, Z_action, cmap=cmap, alpha=0.5)
    plt.colorbar(ctr, ticks = [0,1,2,3,4], format=formatter, ax=ax[1])
    ax[1].set_title("Greedy action" + "\n" + model_name)
    ctr.set_clim(-0.5,4.5)
    ax[1].axis("off")

    if save:
        plot_name = os.path.join(plot_path, f"{run_name}_{rep_name}_{timestep}.png")
        plt.savefig(plot_name)
    plt.show()

