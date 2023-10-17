"""
Main Training Function
"""
from typing import Optional, List, Dict, Sequence
import os, yaml

from stable_baselines3.dqn import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_latest_run_id
import gymnasium as gym
import numpy as np

from algorithm.agent import DualAgent, DQNAgent
from utils.config import extract_config
from envs.dual import DualDrone
from utils import update_nested_dict


def train(
        AgentClass,
        EnvironmentClass,
        config_file: str,
        parent_dir: str,
        run_name: str,
        rep_base_name: Optional[str] = "DQN",
        specific_run_id: Optional[int] = None,
        continue_previous: Optional[bool] = False,
        num_reps: Optional[int] = 1,
        config_overrides: Dict = None,
        log_outputs: Optional[List[str]] = ['csv',],
        progress_bar: Optional[bool] = True,
        verbose: Optional[int] = 1
):
    """
    Train an Agent on an Environment. Supports multiple agents in the same environment.

    Parameters
    ----------
    config_overrides : dict
    AgentClass:
        Class of the agent to instantiate. Must the methods: learn, save, load,
        and either set_logger (for single logger), or
        set_logger_by_dir (for multiple loggers in same run)
    EnvironmentClass:
        Environment in which to train the agent. Must be gym-compatible.
    config_file:
        path to config file to use during training.
    parent_dir:
        Parent folder to group various similar training
    run_name:
        Name for series of similar runs. Usually name of environment.
    rep_base_name:
        Name to use to increment each repetition. Usually algorithm name.
        Default "DQN"
    specific_run_id:
        Instead of adding a new run, continue training in any previous runs.
    continue_previous:
        Continue training in the most recent run.
    num_reps:
        Number of identical repetitions.
    log_outputs:
        List of output types to be used in logging training from stable baselines'
        logger. Currently only 'csv', and 'stdout' supported.
        Default ['csv']
    progress_bar:
        Whether to show progress bar while training. Defaults to showing if
        verbose >= 1.
    verbose:
        Verbosity in regards to outputting information to command line.

    Returns
    -------
    None

    """
    if verbose < 1:
        progress_bar = False
    elif verbose == 1:
        progress_bar = True
    elif verbose >= 2:
        if "stdout" not in log_outputs:
            log_outputs.append("stdout")

    # Allows previous folders to be added
    if specific_run_id is None:
        current_id = get_latest_run_id(os.path.join("logs", parent_dir), run_name)
        if not continue_previous:
            current_id += 1
        current_dir = f"{run_name}_{current_id}"
    else:
        current_dir = f"{run_name}_{specific_run_id}"

    if verbose >= 0:
        print(f"Begin training run name: {current_dir}...")

    # General folder structure to be used for logs, model and config directories
    run_dir = os.path.join(parent_dir, current_dir) # "e.g. test1/MountainCar_7"

    # Create new folders
    for path in ["logs", "model", "config"]:
        # Create upper directory, e.g. "logs/test1"
        if not os.path.exists(os.path.join(path, parent_dir)):
            os.mkdir(os.path.join(path, parent_dir))
        # Create run directories to store run repetitions,
        # e.g. "logs/test1/MountainCar_7"
        if path != "config":
            # Since config only has one file per run
            # It does not need multiple files for each repetition
            joint_path = os.path.join(path, run_dir)
            if not os.path.exists(joint_path):
                os.mkdir(joint_path)

    # Read config and store a copy for current run
    with open(config_file, "r") as file:
        config = yaml.load(file, yaml.SafeLoader)

    if config_overrides is not None:
        if verbose >= 1:
            print(f"Updating config with {config_overrides}")
        config = update_nested_dict(config, config_overrides)

    # Stores a copy of current training session as new file
    new_config_file = os.path.join("config", parent_dir, f"{current_dir}.yaml")
    with open(new_config_file, "w") as file:
        yaml.dump(config, file)

    # Enables continued training in same directory
    for rep in range(num_reps):
        rep_id = get_latest_run_id(os.path.join("logs", run_dir), rep_base_name) + 1
        rep_name = f"{rep_base_name}_{rep_id}"
        rep_path = os.path.join(run_dir, rep_name)

        if verbose >= 1:
            print(f"Starting training on Repetition: {rep_name}...")

        env = EnvironmentClass(
            verbose = verbose,
            **config["environment"]
        )

        agent = AgentClass(
            env,
            verbose=verbose,
            **config["agent"]
        )

        log_dir = os.path.join("logs", rep_path)
        # print(f"Logging to {log_dir}")
        if "set_logger_by_dir" in AgentClass.__dict__:
            agent.set_logger_by_dir(log_dir, format_strings=log_outputs)
        elif "set_logger" in AgentClass.__dict__:
            logger = configure(
                log_dir,
                format_strings=log_outputs
            )
            agent.set_logger(logger)
        else:
            agent.set_logger_by_dir(log_dir, format_strings=log_outputs)

        agent.learn(
            progress_bar=progress_bar,
            **config["learn"]
        )

        # Saving model
        model_dir = os.path.join("model",run_dir)

        agent.save(model_dir, rep_name)

        if verbose >= 1:
            print(f"Finished training on Repetition: {rep_name}!")