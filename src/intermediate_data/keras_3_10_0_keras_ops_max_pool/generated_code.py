import keras
import numpy as np

def call_func(inputs, pool_size, strides=None, padding="valid", data_format=None):
    output = keras.ops.max_pool(inputs, pool_size, strides, padding, data_format)
    return output

input_tensor = keras.random.normal((2, 5, 5, 3))
example_output = call_func(input_tensor, pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_last")