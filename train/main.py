"""
Main training script.

Meant to be edited.
"""
import datetime

from train import train
from envs import DualLateral, MultiDrone
from algorithm import DualAgent, MultiAgent
from utils import extract_config, expand_dict

if __name__ == "__main__":
    ##########
    # Training Details
    parent_dir = "multi1"
    run_name = "DoublePredator"
    experiment_file = None
    config_file = "config/multi1/default.yaml"
    num_reps = 1
    verbose = 1
    continue_run = False
    ##########

    if experiment_file is not None:
        experiment_list = expand_dict(extract_config(experiment_file))

        for config in experiment_list:
            print(datetime.datetime.now())
            train(
                MultiAgent,
                MultiDrone,
                config_file=config_file,
                parent_dir=parent_dir,
                run_name=run_name,
                num_reps=num_reps,
                config_overrides=config,
                verbose=verbose,
                continue_previous=continue_run
            )
            print()
        print(datetime.datetime.now())

    else:
        print(datetime.datetime.now())
        train(
            MultiAgent,
            MultiDrone,
            config_file=config_file,
            parent_dir=parent_dir,
            run_name=run_name,
            num_reps=num_reps,
            verbose=verbose,
            continue_previous=continue_run
        )
        print()
        print(datetime.datetime.now())