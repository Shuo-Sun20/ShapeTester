import tensorflow as tf

def call_func(inputs, approximate=False, name=None):
    return tf.nn.gelu(features=inputs, approximate=approximate, name=name)

example_input = tf.random.normal(shape=(2, 3))
example_output = call_func(inputs=example_input, approximate=True)