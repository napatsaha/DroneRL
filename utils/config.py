import os, yaml


def extract_config(path=None,
                   parent_dir=None,
                   run_name=None,
                   name=None):
    if run_name is not None and parent_dir is not None:
        path = os.path.join("config", parent_dir, f"{run_name}.yaml")

    with open(path, "r") as file:
        config = yaml.load(file, yaml.SafeLoader)

    if name is not None:
        if isinstance(name, str) or len(name) == 1:
            return config[name]
        elif isinstance(name, list) and len(name) > 1:
            dct = config
            for key in name:
                dct = dct[key]
            return dct
    else:
        return config

