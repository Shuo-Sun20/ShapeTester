import tensorflow as tf

def call_func(inputs, name="is_non_decreasing"):
    return tf.math.is_non_decreasing(x=inputs, name=name)

x = tf.random.uniform(shape=(5,), minval=0, maxval=10, dtype=tf.float32)
example_output = call_func(x)