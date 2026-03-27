import tensorflow as tf

def call_func(inputs, axis=-1, name=None):
    return tf.math.softmax(logits=inputs, axis=axis, name=name)

example_input = tf.random.normal(shape=(3, 4))
example_output = call_func(inputs=example_input)