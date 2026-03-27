import keras
import numpy as np

def call_func(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1
):
    return keras.ops.depthwise_conv(
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
channels = 4
kernel_height = 3
kernel_width = 3
depth_multiplier = 1

input_tensor = keras.random.normal((batch_size, height, width, channels))
kernel_tensor = keras.random.normal((kernel_height, kernel_width, channels, depth_multiplier))

example_output = call_func(
    inputs=input_tensor,
    kernel=kernel_tensor,
    strides=2,
    padding="same",
    data_format="channels_last",
    dilation_rate=1
)