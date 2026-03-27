import tensorflow as tf

def call_func(inputs, output_shape, strides, padding, data_format='NWC', dilations=1, name=None):
    input_tensor = inputs[0]
    filters = inputs[1]
    return tf.nn.conv1d_transpose(input=input_tensor, filters=filters, output_shape=output_shape, strides=strides, padding=padding, data_format=data_format, dilations=dilations, name=name)

batch = 2
in_width = 5
in_channels = 3
output_channels = 4
filter_width = 3
strides = 2
padding = "SAME"
data_format = "NWC"

input_tensor = tf.random.normal([batch, in_width, in_channels])
filters = tf.random.normal([filter_width, output_channels, in_channels])
output_width = in_width * strides
output_shape = tf.constant([batch, output_width, output_channels])
example_output = call_func(inputs=[input_tensor, filters], output_shape=output_shape, strides=strides, padding=padding, data_format=data_format)