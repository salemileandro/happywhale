import os
import shutil

import pandas as pd
import numpy as np
import yaml
from typing import List

from happy.config import Config

__all__ = ["print_class_statistics", "clean_mlflow_trash", "read_params"]


def print_class_statistics(df: pd.DataFrame, col: str):
    """
    Print basic statistics on the column `col` DataFrame `df`
    :param pd.DataFrame df: DataFrame that contains the data
    :param str col: Target column on which statistics are computed
    """

    labels, counts = np.unique(df[col], return_counts=True)

    labels = labels[np.flip(np.argsort(counts))]
    counts = counts[np.flip(np.argsort(counts))]

    s = "CLASS NAME".ljust(30) + "COUNTS".ljust(10) + "PERCENTAGE".ljust(10)
    print(s)
    print("-".ljust(50, "-"))
    for c, l in zip(counts, labels):
        s = f"{l}".ljust(30)
        s += f"{c}".ljust(10)
        s += f"{100 * c/ np.sum(counts):.2f}".ljust(10)
        print(s)


def clean_mlflow_trash():
    """
    Delete all contents within the `.trash` folder in the MLflow folder.
    """
    trash_path = os.path.join(Config.MLFLOW_URI, ".trash")
    for i in os.listdir(trash_path):
        dpath = os.path.join(trash_path, i)
        shutil.rmtree(dpath, ignore_errors=True)


def read_params(file: str = None, allow_dict: List[str] = None):
    """
    Read the parameter file that has been created for DVC runs. By default the parameters read are such that they do
    not belong to a nested dict structure. To allow the capture of parameter within a nested dict structure, populate
    the `allow_dict` accordingly.

    :param str file: Path to param file. If None, file is set to `os.path.join(Config.PROJECT_ROOT_DIR, "params.yaml")`.
        Default is None.
    :param List[str] allow_dict: List of str containing the allowed nested dictionary to be considered. Default is None.
    :return: dict with relevant parameters
    """
    # Set default
    if file is None:
        file = os.path.join(Config.PROJECT_ROOT_DIR, "params.yaml")

    # Read parameters form yaml file
    with open(file, 'r') as f:
        yaml_params = yaml.safe_load(f)

    params = {}
    for k, v in yaml_params.items():
        if isinstance(v, dict):
            if isinstance(allow_dict, list) and k in allow_dict:
                for i, j in v.items():
                    params[i] = j
        else:
            params[k] = v

    return params



