import tensorflow as tf

def call_func(inputs, name=None):
    return tf.math.lbeta(x=inputs, name=name)

random_tensor = tf.random.uniform(shape=[2, 3, 4], minval=0.1, maxval=5.0, dtype=tf.float32)
example_output = call_func(inputs=random_tensor)