import os, yaml, itertools
from typing import Dict

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


def update_nested_dict(old_dict: Dict, new_dict: Dict):
    for key, val in new_dict.items():
        if isinstance(val, Dict):
            old_dict[key] = update_nested_dict(old_dict.get(key, {}), val)
        else:
            old_dict[key] = new_dict[key]
    return old_dict


def permute_dict(d):
    key, val = zip(*d.items())
    return [dict(zip(key, v)) for v in itertools.product(*val)]

def dict_of_list_to_list_of_dict(dct):
    for key, val in dct.items():
        if isinstance(val, dict):
            if any(map(lambda x: isinstance(x, list), val.values())):
                return permute_dict(val)
            else:
                l = dict_of_list_to_list_of_dict(val)
                return [{key: item} for item in l]
        elif isinstance(val, list):
            return [{key: item} for item in val]
        else:
            return [dct]

def expand_dict(dct):
    for key, val in dct.items():
        if isinstance(val, dict):
            l = expand_dict(val)
            return [{key: item} for item in l]
        elif isinstance(val, list):
            return [{key: item} for item in val]
        else:
            return [dct]