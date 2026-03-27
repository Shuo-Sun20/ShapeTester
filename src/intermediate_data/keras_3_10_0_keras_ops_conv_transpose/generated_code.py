import keras
import numpy as np

def call_func(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1
):
    return keras.ops.conv_transpose(
        inputs,
        kernel,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate
    )

# Construct valid input tensors
batch_size = 2
input_channels = 3
output_channels = 4
spatial_shape = (5, 5)

inputs = keras.random.normal((batch_size,) + spatial_shape + (input_channels,))
kernel = keras.random.normal((3, 3, output_channels, input_channels))
strides = 2
padding = "same"

example_output = call_func(
    inputs=inputs,
    kernel=kernel,
    strides=strides,
    padding=padding,
    data_format="channels_last"
)