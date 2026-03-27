import tensorflow as tf
import numpy as np

def call_func(inputs, stride, padding, data_format="NWC", dilations=1, name=None):
    input_tensor, filters_tensor = inputs[0], inputs[1]
    return tf.nn.conv1d(input=input_tensor, filters=filters_tensor, stride=stride, padding=padding, data_format=data_format, dilations=dilations, name=name)

batch_size = 2
in_width = 10
in_channels = 3
filter_width = 4
out_channels = 5

input_tensor = tf.random.normal(shape=[batch_size, in_width, in_channels])
filters_tensor = tf.random.normal(shape=[filter_width, in_channels, out_channels])

example_output = call_func(inputs=[input_tensor, filters_tensor], stride=1, padding="VALID")