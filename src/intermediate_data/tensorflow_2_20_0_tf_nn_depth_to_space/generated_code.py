import tensorflow as tf

def call_func(inputs, block_size, data_format="NHWC", name=None):
    return tf.nn.depth_to_space(input=inputs, block_size=block_size, data_format=data_format, name=name)

input_tensor = tf.random.normal(shape=[1, 2, 2, 4])
example_output = call_func(inputs=input_tensor, block_size=2, data_format="NHWC")