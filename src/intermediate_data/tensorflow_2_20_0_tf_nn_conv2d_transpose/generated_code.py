import tensorflow as tf
import numpy as np

def call_func(inputs, output_shape, strides, padding, data_format='NHWC', dilations=None, name=None):
    # Split the combined inputs list into individual tensors
    input_tensor, filters_tensor = inputs[0], inputs[1]
    
    # Call the conv2d_transpose API with all provided parameters
    result = tf.nn.conv2d_transpose(
        input=input_tensor,
        filters=filters_tensor,
        output_shape=output_shape,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )
    return result

# Generate random input tensors for conv2d_transpose
batch_size = 2
height = 4
width = 4
in_channels = 3
out_channels = 5
kernel_h = 3
kernel_w = 3

# Input tensor (NHWC format)
input_tensor = tf.random.normal(shape=[batch_size, height, width, in_channels])

# Filters tensor (height, width, output_channels, in_channels)
filters_tensor = tf.random.normal(shape=[kernel_h, kernel_w, out_channels, in_channels])

# Output shape for transpose convolution (NHWC format)
output_h = height * 2  # Example: stride 2 doubles spatial dimensions
output_w = width * 2
output_shape = tf.constant([batch_size, output_h, output_w, out_channels], dtype=tf.int32)

# Strides for transpose convolution
strides = [1, 2, 2, 1]

# Padding mode
padding = 'SAME'

# Call the function with combined inputs
combined_inputs = [input_tensor, filters_tensor]
example_output = call_func(
    inputs=combined_inputs,
    output_shape=output_shape,
    strides=strides,
    padding=padding,
    data_format='NHWC'
)