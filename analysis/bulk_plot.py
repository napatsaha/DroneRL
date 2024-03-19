"""
Plot a bunch of State Value plots from available models in the specified run_name
"""
from typing import Iterator

import numpy as np

from plotting.plot_state_value import plot_state_value
import os, glob, re
import pandas as pd


def find_available_models(path, rep_base_name) -> pd.DataFrame:
    """
    Returns an Iterator that returns a tuple of (rep_id, timestep) for all available models
    in the current path.

    Parameters
    ----------
    path :
    rep_base_name :

    Returns
    -------
    Iterator

    """

    pattern = re.compile(f"{rep_base_name}_([0-9]+)_[a-z0-9]+(?:_(\d+))?\.pt")
    result = []
    #     pd.DataFrame({
    #     'rep_id': pd.Series(dtype='int'),
    #     'timestep': pd.Series(dtype='int')
    # })

    for file_path in glob.glob(os.path.join(path, f"{rep_base_name}_[0-9]*")):
        res = pattern.findall(file_path)
        id, ts = res[0]
        # print(id, ts)
        ts = None if len(ts) == 0 else int(ts)
        result.append([int(id), ts])

    # Sort index
    result = pd.DataFrame(result, columns=["rep_id", "timestep"])
    result.sort_values(by=["rep_id", "timestep"], inplace=True)
    # Fix NaN to be None, for easier passing into plot_state_value()
    result.replace(np.nan, None, inplace=True)

    return result
    # return result.itertuples(index=False, name=None)


if __name__ == "__main__":
    # Configurations
    parent_dir = "test2"
    run_base_name = "TestWorked"
    run_id = 1
    rep_base_name = "DQN"
    save = True
    show = False

    models_available = find_available_models(
        os.path.join("model", parent_dir, f"{run_base_name}_{run_id}"),
        rep_base_name
    )

    # Loop through available models
    iterator = models_available.itertuples(index=False, name=None)
    for rep_id, timestep in iterator:

        rep_name = f"{rep_base_name}_{rep_id}"

        plot_state_value(
            parent_dir,
            run_base_name,
            run_id,
            rep_name,
            timestep=timestep,
            save=save,
            show=show,
            no_timestep_behaviour="final"
        )