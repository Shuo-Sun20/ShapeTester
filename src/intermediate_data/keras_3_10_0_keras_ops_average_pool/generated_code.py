import keras
import numpy as np

def call_func(inputs, pool_size, strides=None, padding="valid", data_format=None):
    return keras.ops.average_pool(inputs, pool_size, strides, padding, data_format)

# Generate random input tensor for channels_last format (batch, height, width, channels)
batch_size, height, width, channels = 2, 8, 8, 3
input_tensor = keras.random.normal(shape=(batch_size, height, width, channels))
pool_size = (2, 2)
strides = (2, 2)

example_output = call_func(input_tensor, pool_size, strides, padding="valid", data_format="channels_last")