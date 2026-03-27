import tensorflow as tf

def call_func(inputs, window_shape, pooling_type, strides=None, padding="SAME", data_format=None, dilations=None, name=None):
    return tf.nn.pool(input=inputs, window_shape=window_shape, pooling_type=pooling_type, strides=strides, padding=padding, data_format=data_format, dilations=dilations, name=name)

example_output = call_func(
    inputs=tf.random.normal(shape=[4, 32, 32, 3]),  # NHWC format (4 batches, 32x32 spatial, 3 channels)
    window_shape=[2, 2],
    pooling_type='AVG',
    strides=[2, 2],
    padding='VALID',
    data_format='NHWC',
    dilations=[1, 1]
)