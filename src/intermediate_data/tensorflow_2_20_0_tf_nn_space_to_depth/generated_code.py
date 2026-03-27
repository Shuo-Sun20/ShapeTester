import tensorflow as tf

def call_func(inputs, block_size, data_format='NHWC', name=None):
    return tf.nn.space_to_depth(inputs, block_size, data_format, name)

input_tensor = tf.random.uniform(shape=[1, 4, 4, 3], minval=0, maxval=1, dtype=tf.float32)
example_output = call_func(input_tensor, block_size=2)