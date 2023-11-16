import os
from typing import Union, List


def restructure_log_dir(parent_dir, agent_names: Union[str, List[str]] = ("predator"), main_dir = "logs"):
    """
    Restructure parent_dir of experiment from older single-agent API
     to better fit the API of a multi-agent script.

    i.e. 'progress.csv' files inside 'DQN_1/' will be stored in 'DQN_1/predator/' instead.

    This will allow scripts such as 'plot.py' to be able to read these files irrespective of
    being a single-, dual- or multi-agent environment.
    """
    # pass

    walk = os.walk(os.path.join(main_dir, parent_dir))

    for dirpath, dirnames, filenames in walk:
        # if there is a csv file in here
        csv_files = [*filter(lambda x: x.endswith("csv"), filenames)]

        current_dir = os.path.split(dirpath)[-1]
        if len(csv_files) > 0 and current_dir not in agent_names:
            for file, agent in zip(csv_files, agent_names):
                # Prepare new directory
                new_dir = os.path.join(dirpath, agent)
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)

                # Move file
                os.rename(os.path.join(dirpath, file), os.path.join(new_dir, file))

                print(f"Moving {file} from {dirpath} to {new_dir}")

if __name__ == "__main__":
    parent_dir = "colli1"
    agent_names = ["predator"]
    main_dir = "logs"

    restructure_log_dir(parent_dir, agent_names, main_dir)
