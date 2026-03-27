import tensorflow as tf

def call_func(inputs, name=None):
    return tf.nn.elu(features=inputs, name=name)

example_tensor = tf.random.normal(shape=(3, 4), dtype=tf.float32)
example_output = call_func(inputs=example_tensor)