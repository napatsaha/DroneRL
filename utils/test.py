from utils import extract_config, expand_dict

experiment_file = "config/multi1/experiment12-13.yaml"

config = extract_config(experiment_file)

exp_config = expand_dict(config)
