import tensorflow as tf

def call_func(inputs, strides, padding, data_format='NHWC', dilations=[1,1,1,1], name=None):
    input_tensor, filters_tensor = inputs[0], inputs[1]
    return tf.nn.dilation2d(input_tensor, filters_tensor, strides, padding, data_format, dilations, name)

input_tensor = tf.random.normal([2, 5, 5, 3])
filters_tensor = tf.random.normal([3, 3, 3])
example_output = call_func([input_tensor, filters_tensor], [1, 1, 1, 1], 'SAME')