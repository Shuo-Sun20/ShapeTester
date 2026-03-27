import tensorflow as tf

def call_func(inputs, name=None):
    return tf.nn.relu(features=inputs, name=name)

example_input = tf.random.uniform(shape=(5, 5), minval=-1.0, maxval=1.0, dtype=tf.float32)
example_output = call_func(example_input)