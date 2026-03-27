import tensorflow as tf

def call_func(inputs, name=None):
    a, x = inputs[0], inputs[1]
    return tf.math.igammac(a=a, x=x, name=name)

a = tf.random.uniform(shape=(3, 2), minval=0.1, maxval=5.0, dtype=tf.float32)
x = tf.random.uniform(shape=(3, 2), minval=0.1, maxval=5.0, dtype=tf.float32)
example_output = call_func(inputs=[a, x])