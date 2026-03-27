import tensorflow as tf

def call_func(inputs, axis=0, exclusive=False, reverse=False, name=None):
    x = inputs[0] if isinstance(inputs, list) else inputs
    return tf.math.cumprod(x, axis=axis, exclusive=exclusive, reverse=reverse, name=name)

example_input = tf.random.uniform(shape=(3, 4), minval=1, maxval=5, dtype=tf.float32)
example_output = call_func([example_input], axis=1, exclusive=True, reverse=False)