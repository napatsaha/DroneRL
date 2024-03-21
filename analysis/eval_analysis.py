"""
Analyse evaluation from multiple runs (rep_id) of a specific run (run_id)
"""

import os
import pandas as pd


def eval_result(parent_dir,
                run_base_name,
                run_id,
                rep_base_name,
                rep_range):
    def print_diagnostic(df, display_name):
        mean_ep_len = df.loc[:, "ep_len"].mean()
        mean_success = 1 - df.loc[:, "has_truncated"].mean()

        print(f"{display_name}\t"
              f"Episode length: {mean_ep_len:.3f}\t"
              f"Success Rate: {mean_success:.2%}")

    for rep_id in rep_range:
        rep_name = f"{rep_base_name}_{rep_id}"
        filepath = os.path.join("logs", parent_dir, f"{run_base_name}_{run_id}", rep_name, "eval_result.csv")
        if os.path.exists(filepath):
            result = pd.read_csv(filepath)
            print_diagnostic(result, rep_name)


if __name__ == "__main__":
    parent_dir = "test2"
    run_base_name = "TestWorked"
    run_id = 6
    rep_base_name = "DQN"
    rep_range = [*range(1, 11)]

    eval_result(parent_dir,
                run_base_name,
                run_id,
                rep_base_name,
                rep_range)
