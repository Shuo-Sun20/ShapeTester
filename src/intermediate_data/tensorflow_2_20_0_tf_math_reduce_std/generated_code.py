import tensorflow as tf

def call_func(inputs, axis=None, keepdims=False, name=None):
    return tf.math.reduce_std(input_tensor=inputs, axis=axis, keepdims=keepdims, name=name)

random_tensor = tf.random.normal(shape=(3, 4))
example_output = call_func(inputs=random_tensor, axis=0)