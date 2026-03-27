import keras
import numpy as np

def call_func(height, width, seed=None, data_format=None, name=None, dtype=None, inputs=None):
    layer = keras.layers.RandomCrop(
        height=height,
        width=width,
        seed=seed,
        data_format=data_format,
        name=name,
        dtype=dtype
    )
    return layer(inputs)

example_input = keras.random.normal(shape=(4, 256, 256, 3))
example_output = call_func(
    height=224,
    width=224,
    seed=42,
    inputs=example_input
)