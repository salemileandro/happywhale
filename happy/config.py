import os
import mlflow
from subprocess import run
import shutil



x = run("git rev-parse --show-toplevel", shell=True, capture_output=True)
if "fatal: not a git repository" in x.stderr.decode():
    raise NotADirectoryError("Library has to be loaded from a git repo !")

PROJECT_ROOT_DIR = x.stdout.decode().strip()

print(PROJECT_ROOT_DIR)

MLFLOW_URI = os.path.join(PROJECT_ROOT_DIR, "mlruns")
if not os.path.isdir(MLFLOW_URI):
    os.makedirs(MLFLOW_URI, exist_ok=False)
mlflow.set_tracking_uri("file://" + MLFLOW_URI)


def clean_mlflow_trash():
    trash_path = os.path.join(MLFLOW_URI, ".trash")
    for i in os.listdir(trash_path):
        dpath = os.path.join(trash_path, i)
        shutil.rmtree(dpath, ignore_errors=True)








