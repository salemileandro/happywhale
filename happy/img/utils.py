import tensorflow as tf
from typing import Tuple

__all__ = ["jpg_to_numpy"]


def jpg_to_numpy(filename: str, size: Tuple[int, int], method: str ="bicubic", dtype=tf.uint8):
    """
    :param str filename: Path to jpg image.
    :param Tuple[int, int] size: Target size of the array.
    :param dtype: Type of the output numpy array. Default is tf.uint8
    :param str method: Method used for resizing. Default is "bicubic".
    :return: Numpy array of shape [size[0], size[1], 3].
    """
    x = tf.io.read_file(filename)                       # Read image -> Get raw bytes
    x = tf.image.decode_jpeg(x)                         # Decode image -> Get tensor
    x = tf.image.convert_image_dtype(x, tf.float32)
    x = tf.image.resize(x, size, method=method)

    # If img is grayscale, make it RGB
    if x.shape[2] == 1:
        x = tf.image.grayscale_to_rgb(x)

    x = tf.image.convert_image_dtype(x, dtype)

    return x.numpy()


