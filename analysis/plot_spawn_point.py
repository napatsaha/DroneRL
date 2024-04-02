"""
Scatter plot of spawn points from evaluation records
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count

# filename = "logs/test2/TestWorked_7/DQN_10/eval_result.csv"

base_dir = "logs"
parent_dir = "test2"
run_base_name = "TestWorked"
run_id = 7
rep_base_name = "DQN"
file_base_name = "eval_result.csv"

run_name = f"{run_base_name}_{run_id}"

run_collection = []
for rep_id in count(1, 1):
    filename = os.path.join(base_dir, parent_dir, run_name,
                            f"{rep_base_name}_{rep_id}", file_base_name)

    if not os.path.exists(filename):
        break

    rep_result = pd.read_csv(filename)
    run_collection.append(rep_result)

run_result = pd.concat(run_collection)

prey_cols = run_result.columns[run_result.columns.str.startswith("prey")]
pred_cols = run_result.columns[run_result.columns.str.startswith("predator")]



plt.scatter(x=run_result.loc[:, prey_cols].iloc[:, 0],
            y=run_result.loc[:, prey_cols].iloc[:, 1], alpha=0.25)
plt.scatter(x=run_result.loc[:, pred_cols].iloc[:, 0],
            y=run_result.loc[:, pred_cols].iloc[:, 1], alpha=0.25)
plt.xlim(0,1)
plt.ylim(0,1)
plt.title(run_name + " | " + f"{rep_base_name}_1-{rep_id-1}")
plt.show()