import tensorflow as tf

def call_func(inputs, ksize, strides, padding, data_format='NDHWC', name=None):
    return tf.nn.avg_pool3d(inputs[0], ksize, strides, padding, data_format, name)

random_input = tf.random.normal(shape=[2, 5, 7, 9, 4], dtype=tf.float32)
example_output = call_func([random_input], [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'VALID')