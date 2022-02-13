"""
Encode the image for classification
"""
import os
import sys
from subprocess import run
import argparse
from contextlib import redirect_stdout
from multiprocessing import cpu_count

import tensorflow as tf
import tensorflow_hub as hub

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


def image_encoder(backbone: str, freeze: bool = True):
    if backbone == "resnet50":
        m = tf.keras.Sequential([
            tf.keras.Input(shape=[224, 224, 3]),
            hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5")])
    elif backbone == "resnet101":
        m = tf.keras.Sequential([
            tf.keras.Input(shape=[224, 224, 3]),
            hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5")])
    else:
        raise AttributeError("backbone not correct")

    if freeze:
        for layer in m.layers:
            layer.trainable = False

    return m


class ShardedGenerator:
    def __init__(self, df: pd.DataFrame, n_shards: int = None):

        self._df = df.copy()

        self._n_shards = n_shards
        if self._n_shards is None:
            self._n_shards = cpu_count()

        self._df["filepath"] = self._df.apply(lambda x: os.path.join(hp.Config.TRAIN_DIR, x["image"]), axis=1)

    def __call__(self, n):
        with h5.File(os.path.join(hp.Config.PROJECT_ROOT_DIR, "preprocessed_224-224/data.h5"), "r") as f:
            for count, (i, row) in enumerate(self._df.iterrows()):
                if count % self._n_shards != n:
                    continue

                img = tf.convert_to_tensor(f["img"][i])
                img = tf.image.convert_image_dtype(img, tf.float32)

                yield img


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
    # Read params
    params = hp.read_params(os.path.join(hp.Config.PROJECT_ROOT_DIR, "params.yaml"), allow_dict=["encode"])

    params["backbone"] = params.get("backbone", "resnet50")
    params["overwrite"] = params.get("overwrite", "False")
    params["batch"] = params.get("batch", 512)
    params["dst_dir"] = os.path.join(hp.Config.PROJECT_ROOT_DIR, "encoded")
    params["h5_encoded"] = os.path.join(params["dst_dir"], "%s.h5" % params["backbone"])

    os.makedirs(params["dst_dir"], exist_ok=True)

    if os.path.isfile(params["h5_encoded"]) and not params["overwrite"]:
        print(f"File {os.path.isfile(params['h5_encoded'])} exists.")
        print(f"Set preprocess.overwrite to True in the param file to allow overwrite.")
        return

    data_df = pd.read_csv(hp.Config.TRAIN_CSV)

    m = image_encoder(params["backbone"])

    n_shards = 16
    gen = ShardedGenerator(data_df, n_shards)
    out_sign = tf.TensorSpec(shape=(224, 224, 3), dtype=tf.uint8)
    ds = tf.data.Dataset.from_tensor_slices(np.arange(n_shards))
    ds = ds.interleave(lambda x: tf.data.Dataset.from_generator(gen, output_signature=out_sign, args=(x,)),
                       cycle_length=n_shards,
                       block_length=1,
                       num_parallel_calls=n_shards,
                       deterministic=True)

    y = None
    for i in tqdm(ds.batch(params["batch"]), total=np.ceil(int(len(data_df) / params["batch"]))):
        if y is None:
            y = m.predict(i)
        else:
            y = np.vstack((y, m.predict(i)))

    with h5.File(params["h5_encoded"], "w") as f:
        f.create_dataset("encoded", data=y, chunks=True)
        f.create_dataset("image", data=data_df["image"].values)
        f.create_dataset("species", data=data_df["species"].values)
        f.create_dataset("individual_id", data=data_df["individual_id"].values)
        f.create_dataset("id", data=data_df.index.values)


if __name__ == "__main__":
    args = parse_command()
    if not args["verbose"]:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                main()
    else:
        main()

