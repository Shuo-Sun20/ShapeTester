import tensorflow as tf
import numpy as np

def call_func(inputs, rate, padding, name=None):
    # Split the inputs list into value and filters
    value, filters = inputs[0], inputs[1]
    
    # Call tf.nn.atrous_conv2d with the provided parameters
    output = tf.nn.atrous_conv2d(value=value, filters=filters, rate=rate, padding=padding, name=name)
    
    return output

# Generate random input tensors
batch_size = 2
in_height = 32
in_width = 32
in_channels = 3
out_channels = 16
filter_height = 3
filter_width = 3
rate = 2

# Create random value tensor (input image)
value = tf.constant(
    np.random.randn(batch_size, in_height, in_width, in_channels).astype(np.float32)
)

# Create random filters tensor
filters = tf.constant(
    np.random.randn(filter_height, filter_width, in_channels, out_channels).astype(np.float32)
)

# Call the function with SAME padding
example_output = call_func(
    inputs=[value, filters],
    rate=rate,
    padding='SAME'
)