import keras
import numpy as np

def call_func(
    inputs,
    factor,
    value_range=(0, 255),
    data_format=None,
    seed=None
):
    layer = keras.layers.RandomSaturation(
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    return layer(inputs)

# Create random input tensor
batch_size = 4
height = 32
width = 32
channels = 3
inputs = keras.random.uniform(
    shape=(batch_size, height, width, channels),
    minval=0,
    maxval=255,
    dtype="float32"
)

example_output = call_func(
    inputs=inputs,
    factor=0.2,
    value_range=(0, 255),
    data_format=None,
    seed=None
)