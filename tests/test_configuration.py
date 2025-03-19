# %%
import tensorflow as tf
import logging


# %%
def test_GPU_recognition():
    n_GPUs = len(tf.config.list_physical_devices("GPU"))
    logging.info(f"Number of GPUs detected: {n_GPUs}")
    assert n_GPUs >= 1, "Tensorflow did not find any GPU devices."
