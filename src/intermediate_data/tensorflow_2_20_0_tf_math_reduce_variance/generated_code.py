import tensorflow as tf

def call_func(inputs, axis=None, keepdims=False, name=None):
    return tf.math.reduce_variance(input_tensor=inputs, axis=axis, keepdims=keepdims, name=name)

random_tensor = tf.random.normal(shape=(2, 3))
example_output = call_func(random_tensor, axis=1, keepdims=True)