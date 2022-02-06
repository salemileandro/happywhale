# Path import

import os
import sys
from subprocess import run
import argparse
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py as h5
import mlflow

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

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
    :return: Dictionary with the command line set values (or defaults)
    """
    my_parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    my_parser.add_argument('--test-run', action='store_true', help="Development flag")
    my_parser.add_argument('--batch', action='store', type=int, default=32, help="Size of batch (frozen backbone)")

    # Arguments for training with frozen backbone
    my_parser.add_argument('--epochs', action='store', type=int, default=10, help="Number of epochs (frozen backbone)")
    my_parser.add_argument('--lr', action='store', type=float, default=1e-3, help="Learning rate (frozen backbone)")

    # Argument for training with unfrozen backbone
    my_parser.add_argument('--ft', action='store_true', help="Flag to activate finetuning")
    my_parser.add_argument('--ft-epochs', action='store', type=int, default=10, help="Number of epochs (finetuning)")
    my_parser.add_argument('--ft-lr', action='store', type=float, default=1e-5, help="Learning rate (finetuning)")

    args = my_parser.parse_args()

    return vars(args)


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

                label = tf.convert_to_tensor(row["class"], dtype=tf.int64)

                yield img, label


def get_dataset(data: pd.DataFrame, n_shards: int = None):

    if n_shards is None:
        n_shards = cpu_count()

    gen = ShardedGenerator(data, n_shards)

    out_sign = (tf.TensorSpec(shape=(224, 224, 3), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.int64))

    ds = tf.data.Dataset.from_tensor_slices(np.arange(n_shards))

    ds = ds.interleave(lambda x: tf.data.Dataset.from_generator(gen, output_signature=out_sign, args=(x,)),
                       cycle_length=n_shards,
                       block_length=1,
                       num_parallel_calls=n_shards,
                       deterministic=True)

    return ds


if __name__ == "__main__":
    params = parse_command()

    # Read train.csv
    data_df = pd.read_csv(hp.Config.TRAIN_CSV)
    if params["test_run"]:
        data_df = data_df.iloc[:1500]

    # Define a cutoff for aggregation of under-represented classes
    species, counts = np.unique(data_df["species"], return_counts=True)
    params["cutoff"] = int(np.floor(np.max(counts) * 0.07))  # Empirical value which gives reasonable aggregate

    print("STATISTICS ON SPECIES:")
    hp.print_class_statistics(data_df, "species")
    print()

    # Aggregation
    unique_species, count_species = np.unique(data_df["species"], return_counts=True)

    unique_species = unique_species[np.argsort(count_species)]
    count_species = count_species[np.argsort(count_species)]

    map_species = {}
    idx = 0
    acc = 0
    name = []
    for i, j in zip(count_species, unique_species):
        acc += i
        name.append(j)
        if acc >= params["cutoff"]:
            for n in name:
                map_species[n] = idx
            idx += 1
            acc = 0
            name = []
    data_df["class"] = data_df.apply(lambda x: map_species[x["species"]], axis=1)

    print("STATISTICS ON CLASS:")
    hp.print_class_statistics(data_df, "class")
    print()

    # 80 / 20 split
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    train_index, val_index = next(skf.split(np.zeros(len(data_df)), data_df["class"].values))

    train_df = data_df.loc[train_index].copy()
    val_df = data_df.loc[val_index].copy()

    print("Stats for train set:")
    hp.print_class_statistics(train_df, "class")
    print()

    print("\nStats for val set:")
    hp.print_class_statistics(val_df, "class")
    print()

    params["steps_per_epoch"] = int(np.ceil(len(train_df) / params["batch"]))

    ds_train = get_dataset(train_df, n_shards=16)
    ds_train = ds_train.batch(params["batch"]).cache()
    ds_train = ds_train.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = get_dataset(val_df, n_shards=16)
    ds_val = ds_val.batch(params["batch"]).cache()
    ds_val = ds_val.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)


    # Class information
    num_classes = len(train_df["class"].unique())
    print("N classes =", num_classes)

    class_weight = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_df["class"])
    class_weight = dict({i: class_weight[i] for i in range(len(class_weight))})
    print("Class weights:", class_weight)

    # Load model
    model_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(224, 224, 3)),
        hub.KerasLayer(model_URL, trainable=False),
        tf.keras.layers.Dense(num_classes, activation='softmax')])


    # Setup mlflow
    exp_name = "classification"
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is not None:
        exp_id = experiment.experiment_id
    else:
        exp_id = mlflow.create_experiment(name=exp_name)

    with mlflow.start_run(experiment_id=exp_id) as run:

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        params["steps_per_epoch"] = int(np.ceil(len(train_df) / params["batch"]))
        history = model.fit(ds_train.repeat(params["epochs"]),
                            validation_data=ds_val,
                            epochs=params["epochs"],
                            steps_per_epoch=params["steps_per_epoch"],
                            class_weight=class_weight,
                            callbacks=[hp.MLflowCallback()],
                            verbose=2)

        mlflow.log_dict(history.history, "history.json")

        # Finetune if necessary
        if params["ft"]:
            for layer in model.layers:
                layer.trainable = True

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["ft-lr"]),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

            history = model.fit(ds_train.repeat(params["ft-epochs"]),
                                validation_data=ds_val,
                                epochs=params["epochs"],
                                steps_per_epoch=params["steps_per_epoch"],
                                class_weight=class_weight,
                                callbacks=[hp.MLflowCallback(prepend="finetune-")],
                                verbose=2)

            mlflow.log_dict(history.history, "history-finetune.json")
        mlflow.log_dict(params, "params.yaml")
