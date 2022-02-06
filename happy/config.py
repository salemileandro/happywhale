import os
from subprocess import run

import mlflow


class Config:
    """
    Static class for storing project wide config parameters. Modifications of the internals of this class must be
    performed with caution !
    """
    # Capture the root project directory via a git command.
    x = run("git rev-parse --show-toplevel", shell=True, capture_output=True)
    if "fatal: not a git repository" in x.stderr.decode():
        raise NotADirectoryError("Library has to be loaded from a git repo !")
    PROJECT_ROOT_DIR = x.stdout.decode().strip()

    # Set the URI of the MLflow directory
    MLFLOW_URI = os.path.join(PROJECT_ROOT_DIR, "mlruns")
    if not os.path.isdir(MLFLOW_URI):
        os.makedirs(MLFLOW_URI, exist_ok=False)
    mlflow.set_tracking_uri("file://" + MLFLOW_URI)

    # Set the path to "train.csv" file
    TRAIN_CSV = os.path.join(PROJECT_ROOT_DIR, "input/happy-whale-and-dolphin/train.csv")

    # Set the path to train directory
    TRAIN_DIR = os.path.join(PROJECT_ROOT_DIR, "input/happy-whale-and-dolphin/train_images")

    def __init__(self):
        raise PermissionError("Config is a static class, not instance allowed.")











