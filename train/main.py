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
    parent_dir = "colli2"
    run_name = "GridWorld"
    experiment_file = None # For comparing various parameters
    config_file = "config/colli2/default.yaml" # Configuration file; will create copy with run_name_run_id
    num_reps = 2 # Number of repetition in this run
    verbose = 1
    continue_run = False # True or False: To continue most recent run
    run_id = None # Specific run to continue (ignores continue_run)
    ##########

    if experiment_file is not None:
        experiment_list = expand_dict(extract_config(experiment_file))

        for config in experiment_list:
            print(datetime.datetime.now())
            train(
                None,
                None,
                config_file=config_file,
                parent_dir=parent_dir,
                run_name=run_name,
                num_reps=num_reps,
                config_overrides=config,
                verbose=verbose,
                continue_previous=continue_run,
                specific_run_id=run_id
            )
            print()
        print(datetime.datetime.now())

    else:
        print(datetime.datetime.now())
        train(
            None,
            None,
            config_file=config_file,
            parent_dir=parent_dir,
            run_name=run_name,
            num_reps=num_reps,
            verbose=verbose,
            continue_previous=continue_run,
            specific_run_id=run_id
        )
        print()
        print(datetime.datetime.now())
