import tensorflow as tf
import numpy as np

def call_func(
    inputs,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    output_dtype=tf.int64,
    include_batch_in_index=False,
    name=None
):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
        
    output, argmax = tf.nn.max_pool_with_argmax(
        input=input_tensor,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        output_dtype=output_dtype,
        include_batch_in_index=include_batch_in_index,
        name=name
    )
    return output

# Generate random input tensor
batch_size = 2
height = 8
width = 8
channels = 3
input_tensor = tf.random.normal(shape=[batch_size, height, width, channels])

# Call the function
example_output = call_func(
    inputs=input_tensor,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='VALID',
    data_format='NHWC',
    output_dtype=tf.int64,
    include_batch_in_index=False
)