import tensorflow as tf

def call_func(inputs, name=None):
    return tf.linalg.det(input=inputs, name=name)

example_input = tf.random.normal(shape=(3, 3), dtype=tf.float32)
example_output = call_func(inputs=example_input)