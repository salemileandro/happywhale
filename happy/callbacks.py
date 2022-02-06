import tensorflow as tf
import mlflow


class MLflowCallback(tf.keras.callbacks.Callback):
    """
    Child class of tf.keras.callbacks.Callback for registering metrics while training of tf.keras models.
    """
    def __init__(self, prepend: str = None):
        self._prepend = prepend if prepend else ""

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k, v in logs.items():
                mlflow.log_metric(key=self._prepend + k, value=v, step=epoch)
