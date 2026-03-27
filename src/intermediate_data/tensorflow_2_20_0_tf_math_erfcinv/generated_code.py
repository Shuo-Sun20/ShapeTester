import tensorflow as tf

def call_func(inputs, name=None):
    return tf.math.erfcinv(x=inputs, name=name)

tf.random.set_seed(42)
random_tensor = tf.random.uniform(shape=(5,), minval=0.0, maxval=2.0, dtype=tf.float32)
example_output = call_func(random_tensor)