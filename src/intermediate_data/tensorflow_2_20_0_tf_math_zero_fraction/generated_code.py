import tensorflow as tf

def call_func(inputs, name=None):
    return tf.math.zero_fraction(value=inputs, name=name)

example_tensor = tf.constant([1.0, 0.0, 3.0, 0.0, 5.0])
example_output = call_func(example_tensor)