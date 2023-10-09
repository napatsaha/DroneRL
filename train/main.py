"""
Main training script.

Meant to be edited.
"""
import datetime

from train import train
from envs.dual import DualLateral
from algorithm.agent import DualAgent
from utils import extract_config, expand_dict

if __name__ == "__main__":
    experiment_file = "config/dual2/experiment2.yaml"
    experiment_list = expand_dict(extract_config(experiment_file))

    for config in experiment_list:
        print(datetime.datetime.now())
        train(
            DualAgent,
            DualLateral,
            config_file="config/dual2/default.yaml",
            parent_dir="dual2",
            run_name="DualLateral",
            num_reps=10,
            config_overrides=config,
            verbose=1,
            continue_previous=False
        )
        print()
    print(datetime.datetime.now())