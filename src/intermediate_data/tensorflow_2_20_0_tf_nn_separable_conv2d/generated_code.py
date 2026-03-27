import tensorflow as tf

def call_func(
    inputs,
    strides,
    padding,
    data_format='NHWC',
    dilations=None,
    name=None
):
    input_tensor, depthwise_filter, pointwise_filter = inputs
    
    return tf.nn.separable_conv2d(
        input=input_tensor,
        depthwise_filter=depthwise_filter,
        pointwise_filter=pointwise_filter,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )

batch_size = 2
in_height = 5
in_width = 5
in_channels = 3
channel_multiplier = 2
out_channels = 4
filter_height = 3
filter_width = 3

input_tensor = tf.random.normal([batch_size, in_height, in_width, in_channels])
depthwise_filter = tf.random.normal([filter_height, filter_width, in_channels, channel_multiplier])
pointwise_filter = tf.random.normal([1, 1, channel_multiplier * in_channels, out_channels])

strides = [1, 1, 1, 1]
padding = 'SAME'

example_output = call_func(
    inputs=[input_tensor, depthwise_filter, pointwise_filter],
    strides=strides,
    padding=padding,
    data_format='NHWC',
    dilations=None,
    name='separable_conv2d_example'
)