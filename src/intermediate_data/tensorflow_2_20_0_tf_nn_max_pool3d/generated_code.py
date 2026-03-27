import tensorflow as tf
import numpy as np

def call_func(inputs, ksize, strides, padding, data_format='NDHWC', name=None):
    # tf.nn.max_pool3d is a function, not a class
    # Since inputs is specified as a list parameter, we unpack the single input tensor
    input_tensor = inputs[0]
    # Direct API call with all parameters
    output = tf.nn.max_pool3d(
        input=input_tensor,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )
    return output

# Construct a valid input
batch_size = 2
in_depth = 6
in_height = 8
in_width = 8
in_channels = 3
# Generate a random 5D tensor in NDHWC format
input_tensor = tf.constant(
    np.random.randn(batch_size, in_depth, in_height, in_width, in_channels).astype(np.float32)
)
# Define pooling parameters
ksize = [1, 2, 2, 2, 1]  # Pooling window size
strides = [1, 2, 2, 2, 1]  # Strides
padding = 'VALID'
data_format = 'NDHWC'

# Call the function and save output
example_output = call_func(
    inputs=[input_tensor],
    ksize=ksize,
    strides=strides,
    padding=padding,
    data_format=data_format
)