import keras
import numpy as np

def call_func(
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
    name=None,
    inputs=None
):
    layer_instance = keras.layers.AveragePooling3D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )
    return layer_instance(inputs)

# example_input = np.random.rand(2, 30, 30, 30, 3).astype("float32")
# example_output = call_func(
#     pool_size=3,
#     strides=None,
#     padding="valid",
#     data_format=None,
#     name=None,
#     inputs=example_input
# )