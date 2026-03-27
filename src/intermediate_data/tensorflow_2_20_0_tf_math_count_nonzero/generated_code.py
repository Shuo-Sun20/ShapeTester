import tensorflow as tf

def call_func(inputs, axis=None, keepdims=False, dtype=tf.int64, name=None):
    return tf.math.count_nonzero(input=inputs, axis=axis, keepdims=keepdims, dtype=dtype, name=name)

example_input = tf.constant([[0, 1, 0], [1, 1, 0]])
example_output = call_func(inputs=example_input, axis=0)