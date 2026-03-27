import tensorflow as tf

def call_func(inputs, name=None):
    sign, log_abs_determinant = tf.linalg.slogdet(input=inputs, name=name)
    return [sign, log_abs_determinant]

random_tensor = tf.random.normal(shape=(5, 3, 3), dtype=tf.float32)
example_output = call_func(inputs=random_tensor)