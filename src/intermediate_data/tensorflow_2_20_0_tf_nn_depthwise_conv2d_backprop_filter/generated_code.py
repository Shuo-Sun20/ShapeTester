import tensorflow as tf
import numpy as np

def call_func(inputs, filter_sizes, strides, padding, data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
    input_tensor, out_backprop = inputs[0], inputs[1]
    return tf.nn.depthwise_conv2d_backprop_filter(
        input=input_tensor,
        filter_sizes=filter_sizes,
        out_backprop=out_backprop,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )

# Generate random input data for NHWC format
batch_size = 2
in_height = 8
in_width = 8
in_channels = 3
depthwise_multiplier = 2

# Input tensor
input_tensor = tf.convert_to_tensor(
    np.random.randn(batch_size, in_height, in_width, in_channels).astype(np.float32)
)

# Filter sizes
filter_height = 3
filter_width = 3
filter_sizes = tf.constant([filter_height, filter_width, in_channels, depthwise_multiplier], dtype=tf.int32)

# Strides and padding
strides = [1, 2, 2, 1]
padding = "SAME"

# Calculate output shape for same padding with stride 2
out_height = int(np.ceil(in_height / strides[1]))
out_width = int(np.ceil(in_width / strides[2]))
out_channels = in_channels * depthwise_multiplier

# Generate random gradient tensor (out_backprop)
out_backprop = tf.convert_to_tensor(
    np.random.randn(batch_size, out_height, out_width, out_channels).astype(np.float32)
)

# Call the function
inputs_list = [input_tensor, out_backprop]
example_output = call_func(
    inputs=inputs_list,
    filter_sizes=filter_sizes,
    strides=strides,
    padding=padding
)