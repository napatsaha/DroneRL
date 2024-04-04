import os
from itertools import count

import numpy as np
import pandas as pd


def clamp(n, minn, maxn):
    """Clamp scalar value between maximum and minimum."""
    return max(min(maxn, n), minn)


def safe_simplify(item):
    """
    Return only first element in a single-item list or array.
    Otherwise return the original item.
    """
    if '__len__' in item.__dir__():
        if len(item) == 1:
            return item[0]
    return item


def identity(a, b):
    """
    Returns the first of the two arguments no matter what.
    Useful as a counterpart to min() and max().
    """
    return a


def read_rep_level(
        base_dir, parent_dir, run_base_name, run_id, rep_base_name,
        file_base_name, *,
        index_col=None,
        max_reps=None) -> pd.DataFrame:
    """
    Combine a csv from all REPS in a single RUN into a single indexed DataFrame.

    Parameters
    ----------
    base_dir :
    parent_dir :
    run_base_name :
    run_id :
    rep_base_name :
    file_base_name :
    index_col :
    max_reps :

    Returns
    -------
    pd.DataFrame
    """

    max_reps = np.Inf if max_reps is None else max_reps

    run_name = f"{run_base_name}_{run_id}"

    run_collection = []
    rep_ids = []
    for rep_id in count(1, 1):
        filename = os.path.join(base_dir, parent_dir, run_name,
                                f"{rep_base_name}_{rep_id}", file_base_name)

        if not os.path.exists(filename) or rep_id > max_reps:
            break

        # Run Identifier for indexing
        rep_ids.append(rep_id)

        # Read file and set index
        rep_result = pd.read_csv(filename, index_col=index_col)
        run_collection.append(rep_result)

    # Merge result from multiple REPS
    index_names = ['rep'] + run_collection[0].index.names
    run_result = pd.concat(run_collection, keys=rep_ids, names=index_names)

    return run_result


def read_run_level(
        base_dir, parent_dir, run_base_name, run_ids, rep_base_name,
        file_base_name, *,
        index_col=None,
        max_reps=None
):
    bulk_collection = []
    for run_id in run_ids:
        run_result = read_rep_level(base_dir, parent_dir, run_base_name, run_id, rep_base_name,
                                    file_base_name, index_col=index_col, max_reps=max_reps)
        bulk_collection.append(run_result)

    # Merge result from multiple RUNS
    index_names = ['run'] + bulk_collection[0].index.names
    bulk_result = pd.concat(bulk_collection, keys=run_ids, names=index_names)

    return bulk_result
