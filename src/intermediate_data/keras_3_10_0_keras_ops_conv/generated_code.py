import keras
import numpy as np

def call_func(inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    return keras.ops.conv(
        inputs=inputs,
        kernel=kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate
    )

batch_size = 2
height = 5
width = 5
channels = 3
output_channels = 4
kernel_size = 3

inputs = np.random.randn(batch_size, height, width, channels).astype(np.float32)
kernel = np.random.randn(kernel_size, kernel_size, channels, output_channels).astype(np.float32)

example_output = call_func(inputs, kernel, strides=1, padding="same", data_format="channels_last")