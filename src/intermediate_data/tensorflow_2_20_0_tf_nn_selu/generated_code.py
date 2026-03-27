import tensorflow as tf

def call_func(inputs, name=None):
    return tf.nn.selu(features=inputs, name=name)

example_input = tf.random.normal(shape=(3, 4))
example_output = call_func(example_input)