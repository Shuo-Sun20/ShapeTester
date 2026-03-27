import tensorflow as tf

def call_func(inputs, strides, padding, data_format='NHWC', dilations=[1, 1, 1, 1], name=None):
    input_sizes, filter_tensor, out_backprop = inputs
    return tf.nn.depthwise_conv2d_backprop_input(
        input_sizes, filter_tensor, out_backprop, strides, padding, data_format, dilations, name
    )

tf.random.set_seed(42)
input_sizes = tf.constant([2, 4, 4, 3], dtype=tf.int32)
filter_tensor = tf.random.normal([3, 3, 3, 2], dtype=tf.float32)
out_backprop = tf.random.normal([2, 4, 4, 6], dtype=tf.float32)
strides = [1, 1, 1, 1]
padding = "SAME"
example_output = call_func([input_sizes, filter_tensor, out_backprop], strides, padding)