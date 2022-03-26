import os
import re
import random
import sys
from datetime import datetime

import torch
import numpy as np


def find_version(experiment_version: str, checkpoint_dir: str, debug: bool = False):
    """[summary]

    Args:
        experiment_version (str): [description]
        checkpoint_dir (str): [description]
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    # Default version number
    version = 0

    if debug:
        version = "debug"
    else:
        for subdir, dirs, files in os.walk(f"{checkpoint_dir}/{experiment_version}"):
            match = re.search(r".*version_([0-9]+)$", subdir)
            if match:
                match_version = int(match.group(1))
                if match_version > version:
                    version = match_version

        version = str(version + 1)

    full_version = experiment_version + "/version_" + str(version)

    return full_version, experiment_version, str(version)


def set_seed(seed):
    """[summary]

    Args:
        seed ([type]): [description]
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic():
    """[summary]
    """
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """[summary]
    """

    def __init__(self, silent=False):
        self.start = datetime.now()
        self.silent = silent
        if not self.silent:
            print(f"Started at {self.start}")

    def time(self):
        end = datetime.now()

        return end - self.start

    def end(self):
        end = datetime.now()

        if not self.silent:
            print(f"Ended at at {end}")


def progressbar(it, prefix="", size=60, file=sys.stdout):
    """A super simple progressbar.

    Args:
        it ([type]): [description]
        prefix (str, optional): [description]. Defaults to "".
        size (int, optional): [description]. Defaults to 60.
        file ([type], optional): [description]. Defaults to sys.stdout.
    """

    def get_n(j):
        return int(size * j / count)

    def show(x, j):
        file.write(f"{prefix} [{'#' * x}{'.' * (size-x)}] {j:0{n_chars}d}/{count}\r")
        file.flush()

    count = len(it)
    cur_n = 0
    n_chars = len(str(count))

    show(0, 0)

    for i, item in enumerate(it):
        yield item
        x = get_n(i + 1)
        if x != cur_n:
            show(x, i + 1)
            cur_n = x

    file.write("\n")
    file.flush()

class HidePrints(object):
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout