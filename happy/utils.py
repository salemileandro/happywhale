import os

import pandas as pd
import numpy as np


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

