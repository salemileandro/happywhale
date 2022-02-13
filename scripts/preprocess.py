"""
Preprocess the image data from raw jpeg images to a dataset stored in a HDF5 object.
"""
import os
import sys
from subprocess import run
import argparse
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import h5py as h5
from tqdm import tqdm

# Custom library in dev
x = run("git rev-parse --show-toplevel", shell=True, capture_output=True)
if "fatal: not a git repository" in x.stderr.decode():
    raise NotADirectoryError("Library has to be loaded from a git repo !")
if x.stdout.decode().strip() not in sys.path:
    sys.path.append(x.stdout.decode().strip())
import happy as hp


def parse_command():
    """
    Parse command line arguments and return a dictionary of those arguments. The arguments can be inspected from the
    source code itself or via the command `python3 path_to_script.py --help`
    """
    my_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    my_parser.add_argument('-v', dest="verbose", action='store_true', help="verbosity level")

    args = my_parser.parse_args()

    return vars(args)


def main():
    params = hp.read_params(os.path.join(hp.Config.PROJECT_ROOT_DIR, "params.yaml"), allow_dict=["preprocess"])

    # Read parameters and set defaults
    params["size"] = params.get("size", [224, 224])
    params["dst_dir"] = os.path.join(hp.Config.PROJECT_ROOT_DIR, "preprocessed_%d-%d" % tuple(params["size"]))
    params["h5_file"] = os.path.join(params["dst_dir"], "data.h5")
    params["overwrite"] = params.get("overwrite", False)

    if not os.path.isdir(params["dst_dir"]):
        os.makedirs(params["dst_dir"], exist_ok=False)

    # If the file exists and NO overwrite, then we return
    if os.path.isfile(params["h5_file"]) and not params["overwrite"]:
        print(f"File {os.path.isfile(params['h5_file'])} exists.")
        print(f"Set preprocess.overwrite to True in the param file to allow overwrite.")
        return

    # Read the csv file
    data_df = pd.read_csv(hp.Config.TRAIN_CSV)

    # Allocate array
    np_data = np.empty(shape=(len(data_df), *params["size"], 3), dtype=np.uint8)

    # Read and resize the jpg images. Store them in the numpy array
    for idx in tqdm(data_df.index.values):
        filename = os.path.join(hp.Config.TRAIN_DIR, data_df.at[idx, "image"])
        np_data[idx] = hp.img.jpg_to_numpy(filename, size=params["size"], dtype=np.uint8)

    # Write the HDF5 file. Use chunks of shape (1, size[0], size[1], size[2]) to improve per image read speed.
    h5_shape = np_data.shape
    with h5.File(params["h5_file"], "w") as f:
        f.create_dataset("img", data=np_data, shape=h5_shape, chunks=(1, *h5_shape[1:]), dtype=np.uint8)

    return


if __name__ == "__main__":
    args = parse_command()
    if not args["verbose"]:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                main()
    else:
        main()






